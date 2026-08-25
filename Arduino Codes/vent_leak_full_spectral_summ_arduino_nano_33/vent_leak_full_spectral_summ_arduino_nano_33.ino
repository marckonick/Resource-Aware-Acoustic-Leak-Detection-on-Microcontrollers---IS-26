/**
 * Arduino Nano 33 BLE Sense — STREAMING spectral-summary CM system
 * ================================================================
 *
 * Rewrite of the batch (record-then-process) sketch with three goals:
 *   1. Fast feature computation  -> radix-2 FFT + single-pass bin reduction.
 *   2. Real-time pipeline        -> ring buffer; recording and feature
 *                                   extraction overlap (no 106 KB clip store).
 *   3. EWMA on the model output  -> causal smoothing of the leak probability.
 *
 * WHY THIS IS STREAMING-SAFE
 * --------------------------
 *   The features are mean (and std) of per-frame spectral summaries. Those
 *   statistics are order-independent, so they can be accumulated online with
 *   Welford's algorithm — the whole clip never has to live in RAM at once.
 *   A PDM ISR fills a small ring buffer; the main loop pulls one HOP at a time,
 *   computes one frame's features, folds them into running mean/M2, and every
 *   FRAMES_PER_DECISION frames emits a feature vector -> normalize -> Invoke ->
 *   EWMA.
 *
 * DEVIATION FROM TRAINING (read this)
 * -----------------------------------
 *   The original used librosa-style center=True (reflect padding) over a fixed
 *   clip. This streaming version uses VALID frames only (no reflect padding).
 *   Over ~107 frames that changes the mean/std negligibly (2 edge frames), but
 *   if you need exact train/deploy parity, extract your *training* features the
 *   same streaming/no-center way, or re-add reflect at the window boundary.
 *
 * FEATURE ORDER (unchanged): [centroid, bandwidth, rolloff, flatness, zcr, rms]
 * Model input is the 6 means, normalized by (dataMean, dataStd).
 * Std is also computed (free, via Welford) and printed — flip USE_STD_INPUT and
 * extend the model + scaler to a 12-D vector if you want to use it.
 */

#include <PDM.h>
#include <math.h>
#include <string.h>
#include <Chirale_TensorFlowLite.h>
#include "model.h"
#include "saved_features.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ----------------------- Configuration -----------------------
#define SAMPLE_RATE          16000
#define FRAME_SIZE           1024          // analysis window (== NFFT)
#define NFFT                 1024          // power of two -> radix-2 FFT
#define HOP_SIZE             512
#define N_FEAT               6             // per-frame features
#define FRAMES_PER_DECISION  40           // ~3.4 s window (matches old clip)

#define EPS                  1e-10f
#define ROLL_PERCENT         0.85f

#define LEAK_CLASS           1             // model output index treated as "leak"
#define EWMA_ALPHA           0.30f         // smoothing factor (0..1), higher = snappier
#define EWMA_THRESHOLD       0.50f         // decision threshold on smoothed p(leak)

#define USE_STD_INPUT        0             // 0 = feed 6 means (matches current model)
#define DEBUG_DUMP           0             // 1 = dump raw frames over Serial (slow)
#define DEBUG_PRINT          1

static const int K = NFFT / 2 + 1;         // 513 rfft bins

// Ring buffer: power of two so index masking is a single AND.
#define RING_SIZE            4096           // >> FRAME + PDM burst
static const int RING_MASK = RING_SIZE - 1;

// ── Buffers ─────────────────────────────────────────────────────────────────
static volatile int16_t audioRing[RING_SIZE];
static volatile uint32_t ringWrite = 0;    // ISR-owned write cursor
static uint32_t          ringRead  = 0;    // loop-owned read cursor

static float frameBuf[FRAME_SIZE];         // current (overlapping) frame, float
static float fftRe[NFFT];                  // FFT working buffers
static float fftIm[NFFT];
static float mag[K];                       // magnitude spectrum (for rolloff pass)

static float hann[FRAME_SIZE];             // symmetric Hann (np.hanning)
static float freqs[K];                     // precomputed bin frequencies
static float twCos[NFFT / 2];              // FFT twiddles:  cos(2*pi*k/N)
static float twSin[NFFT / 2];              //              -sin(2*pi*k/N)

// Welford running stats over frames (per feature)
static float  wMean[N_FEAT];
static float  wM2[N_FEAT];
static int    wCount = 0;

// EWMA state
static float  ewma = 0.0f;
static bool   ewmaInit = true; // false true

// Frame priming
static int    hopsLoaded = 0;

//time
unsigned long t0;
unsigned long dt; 
static uint64_t totalFrameComputeUs = 0;


// ── AI namespace ────────────────────────────────────────────────────────────
namespace {
const tflite::Model*      model       = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor*             input       = nullptr;
TfLiteTensor*             output      = nullptr;

constexpr int kTensorArenaSize = 4000;     // small MLP; bump if AllocateTensors fails
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

/* ============================ setup helpers ============================ */

void build_tables() {
  for (int n = 0; n < FRAME_SIZE; n++)
    hann[n] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * n / (FRAME_SIZE - 1)));

  for (int k = 0; k < K; k++)
    freqs[k] = (float)k * SAMPLE_RATE / NFFT;

  for (int k = 0; k < NFFT / 2; k++) {
    float ang = 2.0f * (float)M_PI * k / NFFT;
    twCos[k] =  cosf(ang);                 // forward transform: W = cos - i sin
    twSin[k] = -sinf(ang);
  }
}

/* ============================ FFT ============================ */
//
// In-place iterative radix-2 Cooley-Tukey (decimation-in-time).
// Only magnitude is used downstream, so the sign convention is irrelevant.
// SWAP POINT: replace this + the windowing with CMSIS-DSP arm_rfft_fast_f32
// for another ~2-3x on the transform if you set up the CMSIS-DSP library.
void fft_radix2(float* re, float* im) {
  const int n = NFFT;

  // bit-reversal permutation
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = re[i]; re[i] = re[j]; re[j] = tr;
      float ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }

  for (int len = 2; len <= n; len <<= 1) {
    int half = len >> 1;
    int step = n / len;                    // twiddle stride into twCos/twSin
    for (int i = 0; i < n; i += len) {
      int k = 0;
      for (int j = 0; j < half; j++) {
        float wr = twCos[k], wi = twSin[k];
        int a = i + j, b = a + half;
        float ur = re[a], ui = im[a];
        float vr = re[b] * wr - im[b] * wi;
        float vi = re[b] * wi + im[b] * wr;
        re[a] = ur + vr; im[a] = ui + vi;
        re[b] = ur - vr; im[b] = ui - vi;
        k += step;
      }
    }
  }
}

/* ============================ per-frame features ============================ */
//
// Assumes frameBuf[] holds the current raw frame. Windows into fftRe (while
// folding in zcr + rms on the raw samples), runs the FFT, then does ONE heavy
// pass over the bins (mag_sum, f*mag, f^2*mag, power_sum, log_power_sum) plus a
// cheap cumulative pass for rolloff. Writes 6 features to out[].
void compute_frame_features(float* out) {
  // 1. Window + fold in raw-domain zcr & rms in a single pass
  float sumsq = 0.0f;
  int   sign_changes = 0;
  float prev_sign = (frameBuf[0] >= 0.0f) ? 1.0f : -1.0f;

  for (int n = 0; n < FRAME_SIZE; n++) {
    float raw = frameBuf[n];
    sumsq += raw * raw;
    float s = (raw >= 0.0f) ? 1.0f : -1.0f;
    if (n > 0 && s != prev_sign) sign_changes++;
    prev_sign = s;

    fftRe[n] = raw * hann[n];
    fftIm[n] = 0.0f;
  }

  // 2. FFT
  fft_radix2(fftRe, fftIm);

  // 3. Single heavy pass over bins
  float magSum = 0.0f, wSum = 0.0f, w2Sum = 0.0f;
  float powerSum = 0.0f, logPowerSum = 0.0f;
  for (int k = 0; k < K; k++) {
    float re = fftRe[k], im = fftIm[k];
    float power = re * re + im * im;       // no sqrt needed for power terms
    float m = sqrtf(power);
    mag[k] = m;                            // stored for the rolloff pass

    float f = freqs[k];
    magSum      += m;
    wSum        += f * m;
    w2Sum       += f * f * m;
    powerSum    += power;
    logPowerSum += logf(power + EPS);
  }
  float magSumEps = magSum + EPS;

  // centroid
  float centroid = wSum / magSumEps;
  // bandwidth via identity: sum m (f-c)^2 = w2Sum - 2c*wSum + c^2*magSum
  float bwNum = w2Sum - 2.0f * centroid * wSum + centroid * centroid * magSum;
  float bandwidth = sqrtf((bwNum > 0.0f ? bwNum : 0.0f) / magSumEps);
  // flatness = geo_mean(power) / arith_mean(power)
  float flatness = expf(logPowerSum / K) / (powerSum / K + EPS);

  // 4. Rolloff: cheap cumulative pass (threshold known only now)
  float threshold = ROLL_PERCENT * magSum;
  float cumulative = 0.0f;
  int rolloff_idx = 0;
  for (int k = 0; k < K; k++) {
    cumulative += mag[k];
    if (cumulative >= threshold) { rolloff_idx = k; break; }
  }
  float rolloff = freqs[rolloff_idx];

  float zcr = (float)sign_changes / (2.0f * FRAME_SIZE);
  float rms = sqrtf(sumsq / FRAME_SIZE);

  out[0] = centroid;
  out[1] = bandwidth;
  out[2] = rolloff;
  out[3] = flatness;
  out[4] = zcr;
  out[5] = rms;
}

/* ============================ Welford running stats ============================ */
void stats_reset() {
  for (int j = 0; j < N_FEAT; j++) { wMean[j] = 0.0f; wM2[j] = 0.0f; }
  wCount = 0;
}

void stats_update(const float* x) {
  wCount++;
  for (int j = 0; j < N_FEAT; j++) {
    float delta = x[j] - wMean[j];
    wMean[j] += delta / wCount;
    float delta2 = x[j] - wMean[j];
    wM2[j] += delta * delta2;
  }
}

// population std (ddof=0), matches np.std default
void stats_finalize(float* meanOut, float* stdOut) {
  for (int j = 0; j < N_FEAT; j++) {
    meanOut[j] = wMean[j];
    stdOut[j]  = (wCount > 0) ? sqrtf(wM2[j] / wCount) : 0.0f;
  }
}

/* ============================ PDM ISR ============================ */
void onPDMdata() {
  static int16_t tmp[256];
  int bytes = PDM.available();
  if (bytes > (int)sizeof(tmp)) bytes = sizeof(tmp);
  PDM.read(tmp, bytes);
  int n = bytes / 2;
  uint32_t w = ringWrite;
  for (int i = 0; i < n; i++) audioRing[(w + i) & RING_MASK] = tmp[i];
  ringWrite = w + n;                       // single volatile publish
}

static inline uint32_t ring_available() { return ringWrite - ringRead; }

// Pull HOP new samples into the tail of frameBuf (after shifting it down).
void load_hop() {
  memmove(frameBuf, frameBuf + HOP_SIZE, (FRAME_SIZE - HOP_SIZE) * sizeof(float));
  for (int i = 0; i < HOP_SIZE; i++)
    frameBuf[FRAME_SIZE - HOP_SIZE + i] = (float)audioRing[(ringRead + i) & RING_MASK];
  ringRead += HOP_SIZE;
}

/* ============================ inference + EWMA ============================ */
void run_inference(const float* mean, const float* stdv) {
  // Normalize into the model input (6-D means by default).
  for (int i = 0; i < N_FEAT; i++) {
    float v = (float)((mean[i] - dataMean[i]) / dataStd[i]);
    input->data.f[i] = v;
#if USE_STD_INPUT
    // 12-D variant: also feed normalized std. Requires a 12-D model and
    // 12-element dataMean/dataStd. Left here as the switch to flip.
    input->data.f[N_FEAT + i] = (float)((stdv[i] - dataMean[N_FEAT + i]) / dataStd[N_FEAT + i]);
#endif
  }

  if (interpreter->Invoke() != kTfLiteOk) { MicroPrintf("Invoke failed"); return; }

  // Softmax over the 2 outputs -> probability (matches CrossEntropyLoss training,
  // and gives EWMA a calibrated quantity to smooth).
  float o0 = output->data.f[0], o1 = output->data.f[1];
  float mx = (o0 > o1) ? o0 : o1;
  float e0 = expf(o0 - mx), e1 = expf(o1 - mx);
  float probs[2] = { e0 / (e0 + e1), e1 / (e0 + e1) };
  float pLeak = probs[LEAK_CLASS];

  // Causal EWMA
  if (!ewmaInit) { ewma = pLeak; ewmaInit = true; }
  else           { ewma = EWMA_ALPHA * pLeak + (1.0f - EWMA_ALPHA) * ewma; }

  bool leak = (ewma >= EWMA_THRESHOLD);

  Serial.print("p(leak)="); Serial.print(pLeak, 4);
  Serial.print("  ewma=");  Serial.print(ewma, 4);
  Serial.print("  -> ");    Serial.print(leak ? "LEAK" : "OK");
  
  //Serial.print("  (invoke us="); Serial.print(dt); Serial.println(")");

  digitalWrite(LEDR, leak ? LOW : HIGH);   // red ON = leak
}

/* ============================ Arduino entry points ============================ */
void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}   // headless-safe (no infinite wait)

  pinMode(LEDR, OUTPUT); digitalWrite(LEDR, HIGH);
  pinMode(LEDG, OUTPUT); digitalWrite(LEDG, HIGH);
  pinMode(LEDB, OUTPUT); digitalWrite(LEDB, HIGH);

  build_tables();
  stats_reset();

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d != %d", model->version(), TFLITE_SCHEMA_VERSION);
    while (true) {}
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed");
    while (true) {}
  }
  input  = interpreter->input(0);
  output = interpreter->output(0);

  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("ERROR: PDM init failed");
    while (true) {}
  }

  Serial.println("Streaming CM running (continuous).");
  digitalWrite(LEDB, LOW);                 // blue ON = listening
  t0 = micros();
}

void loop() {
  // Consume one HOP whenever it's available; everything else is event-driven.
  if (ring_available() < (uint32_t)HOP_SIZE) return;

  load_hop(); // uzme 512 semplova ako ima -> update za frameBuf

  // Prime: need FRAME_SIZE/HOP_SIZE hops before the first frame is fully valid.
  if (hopsLoaded < (FRAME_SIZE / HOP_SIZE)) { hopsLoaded++; return; }

  float feat[N_FEAT];
  unsigned long t0 = micros();
  compute_frame_features(feat); // FEATURE EXTRACTION 
  stats_update(feat);
  totalFrameComputeUs += micros() - t0;

#if  DEBUG_DUMP// this is no 
  Serial.print("frame feats:");
  for (int j = 0; j < N_FEAT; j++) { Serial.print(' '); Serial.print(feat[j], 4); }
  Serial.println();
#endif

if (wCount >= FRAMES_PER_DECISION) {
    float mean[N_FEAT], stdv[N_FEAT];
    unsigned long tFinalize0 = micros();
    stats_finalize(mean, stdv);
    unsigned long finalizeUs = micros() - tFinalize0;
   
  t0 = micros();
  run_inference(mean, stdv);
  dt = micros() - t0;
  
#if DEBUG_PRINT

    Serial.print("== window ("); Serial.print(wCount); Serial.println(" frames) ==");
    const char* names[N_FEAT] = {"centroid","bandwidth","rolloff","flatness","zcr","rms"};
    for (int j = 0; j < N_FEAT; j++) {
      Serial.print("  "); Serial.print(names[j]);
      Serial.print(" mean="); Serial.print(mean[j], 4);
      Serial.print(" std=");  Serial.println(stdv[j], 4);
    }

   unsigned long featUs = totalFrameComputeUs;
   Serial.print(" feature extraction time (us)="); Serial.print(featUs); Serial.println(" ==");
   Serial.print(" Finalization Time (us): "); Serial.print(finalizeUs); Serial.println(" ==");
   Serial.print(" Inference Time (us): "); Serial.print(dt); Serial.println(" ==");
#endif
    stats_reset();                         // start next decision window
    t0 = micros();
    totalFrameComputeUs = 0;
  }
}





