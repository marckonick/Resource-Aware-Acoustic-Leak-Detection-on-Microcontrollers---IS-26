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
 * Band-power feature extractor — Arduino port of Python extract_bandpower().
 * Drop-in replacement for compute_frame_features(): 1024-sample frame -> 8 bands.
 *
 * Reuses from the streaming sketch: fft_radix2(), fftRe[NFFT], fftIm[NFFT],
 * twCos[]/twSin[], and the constants NFFT / FRAME_SIZE / SAMPLE_RATE / K.
 * Set N_FEAT = N_BANDS (8) and call build_bandpower_tables() in setup().
 *
 * Matches Python:
 *   S = |librosa.stft(y, n_fft, hop)|**2            (power spectrum, per frame)
 *   bp[b] = sum of S over bins with freqs in [edge[b], edge[b+1])
 *   if log_power:  bp = log(bp + eps)
 *   if ref_band:   bp = bp - bp[ref_band]           (scale-invariant ratios)
 *
 * IMPORTANT PARITY NOTES
 * ----------------------
 *  1. WINDOW: librosa.stft uses a PERIODIC Hann (0.5*(1-cos(2*pi*n/N))), which
 *     differs from the symmetric np.hanning used in the spectral-summary sketch.
 *     This file uses the periodic one — do NOT reuse the old hann[] table here.
 *  2. BAND EDGES: this assumes _band_edges() is log-spaced == np.geomspace(fmin,
 *     fmax, n_bands+1) with endpoints pinned exactly (that's what numpy does).
 *     If your _band_edges() differs, the bins will differ. The bullet-proof
 *     option is to compute the integer bin ranges ONCE in Python and paste them
 *     in (see snippet at the bottom) — that guarantees exact parity and skips
 *     the on-device edge math entirely.
 *  3. Verified against a NumPy reference: outputs match to ~1e-5, bin ranges
 *     identical including the Nyquist-bin exclusion at the top band.
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
#define N_FEAT               8             // per-frame features
#define FRAMES_PER_DECISION  40           // ~3.4 s window (matches old clip)

#define EPS                  1e-10f
#define ROLL_PERCENT         0.85f

#define LEAK_CLASS           1             // model output index treated as "leak"
#define EWMA_ALPHA           0.30f         // smoothing factor (0..1), higher = snappier
#define EWMA_THRESHOLD       0.50f         // decision threshold on smoothed p(leak)

#define USE_STD_INPUT        0             // 0 = feed 6 means (matches current model)
#define DEBUG_DUMP           0             // 1 = dump raw frames over Serial (slow)
#define DEBUG_PRINT          1

#define N_BANDS        8
#define BP_FMIN        50.0f
#define BP_FMAX        (SAMPLE_RATE / 2.0f)   // librosa default fmax=None -> Nyquist
#define BP_SCALE_LOG   1        // 1 = log-spaced (geomspace), 0 = linear
#define BP_LOG_POWER   1        // 1 = log(power), 0 = linear power
#define BP_REF_BAND   (-1)      // -1 = disabled; else 0..N_BANDS-1 -> band ratios

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

// Periodic Hann (librosa.stft default). Separate from the symmetric hann[].
// Contiguous STFT-bin range [lo, hi) for each band (precomputed once).
static int   bandLo[N_BANDS];
static int   bandHi[N_BANDS];


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

// ----------------------- Setup: call once in setup() -----------------------
void build_bandpower_tables() {

  // Periodic Hann: w[n] = 0.5 * (1 - cos(2*pi*n / N)),  N = n_fft
  for (int n = 0; n < FRAME_SIZE; n++)
    hann[n] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * n / NFFT));

    // FFT twiddles — REQUIRED before any fft_radix2() call.
  for (int k = 0; k < NFFT / 2; k++) {
    float ang = 2.0f * (float)M_PI * k / NFFT;
    twCos[k] =  cosf(ang);   //  cos(2*pi*k/N)
    twSin[k] = -sinf(ang);   // -sin(2*pi*k/N)  (forward transform)
  }

   // Band edges. Endpoints pinned exactly (matches np.geomspace/linspace).
  double edges[N_BANDS + 1];
  edges[0] = BP_FMIN;
  edges[N_BANDS] = BP_FMAX;
#if BP_SCALE_LOG
  double lm = log((double)BP_FMIN), lx = log((double)BP_FMAX);
  for (int b = 1; b < N_BANDS; b++)
    edges[b] = exp(lm + (lx - lm) * (double)b / N_BANDS);
#else
  for (int b = 1; b < N_BANDS; b++)
    edges[b] = BP_FMIN + (BP_FMAX - BP_FMIN) * (double)b / N_BANDS;
#endif

  // Map edges -> contiguous bin ranges using freqs[k] = k*sr/n_fft and the
  // half-open mask (freqs >= lo) & (freqs < hi), exactly as in Python.
  for (int b = 0; b < N_BANDS; b++) {
    int lo = K, hi = 0;                       // empty band default -> loop skips
    for (int k = 0; k < K; k++) {
      double f = (double)k * SAMPLE_RATE / NFFT;
      if (f >= edges[b] && f < edges[b + 1]) { if (k < lo) lo = k; if (k + 1 > hi) hi = k + 1; }
    }
    bandLo[b] = lo;
    bandHi[b] = hi;
  }
}

/* ============================ per-frame features ============================ */
// ----------------------- Per-frame feature: 1024 samples -> N_BANDS -----------------------
void compute_frame_bandpower(float* out) {
  // Window with the periodic Hann, load into FFT buffers.
  for (int n = 0; n < FRAME_SIZE; n++) {
    fftRe[n] = frameBuf[n] * hann[n];
    fftIm[n] = 0.0f;
  }

  fft_radix2(fftRe, fftIm);

  // Sum |X[k]|^2 over each band's bins (power spectrum).
  for (int b = 0; b < N_BANDS; b++) {
    float acc = 0.0f;
    for (int k = bandLo[b]; k < bandHi[b]; k++) {
      float re = fftRe[k], im = fftIm[k];
      acc += re * re + im * im;               // power
    }
#if BP_LOG_POWER
    out[b] = logf(acc + EPS);              // empty band -> log(eps), matches Python
#else
    out[b] = acc;
#endif
  }

#if BP_LOG_POWER && (BP_REF_BAND >= 0)
  float ref = out[BP_REF_BAND];
  for (int b = 0; b < N_BANDS; b++) out[b] -= ref;   // scale-invariant ratios
#endif
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

  build_bandpower_tables();
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
  //t0 = micros();
}

void loop() {
  // Consume one HOP whenever it's available; everything else is event-driven.
  if (ring_available() < (uint32_t)HOP_SIZE) return;

  load_hop(); // uzme 512 semplova ako ima -> update za frameBuf

  // Prime: need FRAME_SIZE/HOP_SIZE hops before the first frame is fully valid.
  if (hopsLoaded < (FRAME_SIZE / HOP_SIZE)) { hopsLoaded++; return; }

  float feat[N_FEAT];
  unsigned long t0 = micros();
  compute_frame_bandpower(feat);
  stats_update(feat);
  totalFrameComputeUs += micros() - t0;


#if DEBUG_DUMP
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
    const char* names[N_FEAT] = {"1","2","3","4","5","6", "7","8"};
    for (int j = 0; j < N_FEAT; j++) {
      Serial.print("  "); Serial.print(names[j]);
      Serial.print(" mean="); Serial.print(mean[j], 4);
      Serial.print(" std=");  Serial.println(stdv[j], 4);
    }

    unsigned long featUs = totalFrameComputeUs; //micros() - tFeat0;
    Serial.print(" feature extraction time (us)="); Serial.print(featUs); Serial.println(" ==");
    Serial.print(" Finalization time: "); Serial.print(finalizeUs); Serial.println(" ==");
    Serial.print(" Inference time (us): "); Serial.print(dt); Serial.println(" ==");

#endif

    stats_reset();                         // start next decision window

    totalFrameComputeUs = 0;
  }
}




