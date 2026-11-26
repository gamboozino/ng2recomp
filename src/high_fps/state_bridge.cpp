// ng2 - Ninja Gaiden II static recompilation
//
// Phase 3: Deterministic state snapshot bridge & geometry-preserving
// skeletal/pose interpolation.

#include "high_fps/state_bridge.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <rex/logging.h>
#include <rex/runtime.h>
#include <rex/system/xmemory.h>
#include <rex/types.h>

#include "gpu/guest.h"
#include "high_fps/high_fps.h"

namespace ng2::high_fps::state_bridge {
namespace {

std::atomic<uint64_t> g_readableEpoch{1};

bool IsGuestReadable(uint32_t addr, uint32_t size) {
  if (addr < 0x80000000u || uint64_t(addr) + size > 0x100000000ULL) {
    return false;
  }
  const uint32_t page0 = addr >> 12;
  const uint32_t page1 = (size > 1) ? ((addr + size - 1) >> 12) : page0;
  thread_local uint64_t t_epoch = 0;
  thread_local uint32_t t_validPages[64] = {};
  const uint64_t curEpoch = g_readableEpoch.load(std::memory_order_acquire);
  if (t_epoch != curEpoch) {
    std::memset(t_validPages, 0, sizeof(t_validPages));
    t_epoch = curEpoch;
  }
  const uint32_t slot0 = (page0 ^ (page0 >> 6)) & 63u;
  const uint32_t slot1 = (page1 ^ (page1 >> 6)) & 63u;
  if (t_validPages[slot0] == page0 &&
      (page1 == page0 || t_validPages[slot1] == page1)) {
    return true;
  }

  auto* rt = rex::Runtime::instance();
  auto* mem = rt ? rt->memory() : nullptr;
  if (!mem) {
    return false;
  }
  auto* heap = mem->LookupHeap(addr);
  uint32_t protect = 0;
  if (!heap || !heap->QueryProtect(addr, &protect) ||
      !(protect & rex::memory::kMemoryProtectRead)) {
    return false;
  }
  if (size > 1) {
    uint32_t endAddr = addr + size - 1;
    auto* heapEnd = mem->LookupHeap(endAddr);
    uint32_t protEnd = 0;
    if (!heapEnd || !heapEnd->QueryProtect(endAddr, &protEnd) ||
        !(protEnd & rex::memory::kMemoryProtectRead)) {
      return false;
    }
  }
  t_validPages[slot0] = page0;
  if (page1 != page0) {
    t_validPages[slot1] = page1;
  }
  return true;
}

// Guest globals (TU3).
constexpr uint32_t kPoseTablesBase = 0x83AA56C8;
constexpr uint32_t kPoseTable980Offset = 704;   // 0x2C0
constexpr uint32_t kRootCacheTableOffset = 736; // 0x2E0
constexpr uint32_t kPoseTableE00Offset = 1104;  // 0x450
constexpr uint32_t kPoseTableAD0Offset = 1216;  // 0x4C0
constexpr uint32_t kPartParityTableOffset = 688; // 0x2B0
constexpr uint32_t kActiveRenderBufIdx = 0x84C3DA38;

constexpr uint32_t kRenderObjectTable = 0x84C3DC60;
constexpr uint32_t kRenderObjectStride = 24;
constexpr uint32_t kRenderObjectCount = 160;

constexpr uint32_t kCharSlotsBase = 0x8565B4F0;
constexpr uint32_t kCharSlotStride = 0x5B10;
constexpr uint32_t kCharSlotCount = 16;
constexpr uint32_t kCharPosXOffset = 0x270;
constexpr uint32_t kCharHealthOffset = 0x591E;

constexpr uint32_t kBonePaletteArray = 0x83C105D8;
constexpr uint32_t kBonePaletteSlotIdx = 0x84C537F0;
constexpr uint32_t kMaxBoneGroups = 32;
constexpr uint32_t kMaxBonesPerGroup = 256;
constexpr uint32_t kBoneGroupByteStride = 0x3000; // 256 * 48 bytes

// Any displacement above this threshold in a single logic frame is treated as
// a teleport / camera-cut staging jump / pool reuse and snaps to Curr.
constexpr float kTeleportThreshold = 4.0f;
constexpr float kTeleportThresholdSq = kTeleportThreshold * kTeleportThreshold;

// Scale ratio / orthogonality bounds outside which affine decomposition falls
// back cleanly to Curr (prevents shear distortion and dismemberment spikes).
constexpr float kMinBoneScale = 1e-4f;
constexpr float kMaxScaleAnisotropy = 1.12f;
constexpr float kMaxAxisDot = 0.08f;
constexpr float kMinDet = 1e-4f;
constexpr float kMaxScaleJumpRatio = 2.5f;

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct Quat {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 1.0f;
};

struct DecomposedPose {
  bool validAffine = false;
  Quat rot{};
  Vec3 trans{};
  float scale = 1.0f;
  float raw[4][4] = {};
};

inline float Dot3(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 Cross3(const Vec3& a, const Vec3& b) {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

inline float LengthSq3(const Vec3& a) { return Dot3(a, a); }
inline float Length3(const Vec3& a) { return std::sqrt(LengthSq3(a)); }

inline Vec3 Normalize3(const Vec3& a) {
  float len = Length3(a);
  if (len <= 1e-12f) return {0.0f, 0.0f, 0.0f};
  float inv = 1.0f / len;
  return {a.x * inv, a.y * inv, a.z * inv};
}

inline Quat NormalizeQuat(const Quat& q) {
  float n2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (n2 <= 1e-12f || !std::isfinite(n2)) return {0.0f, 0.0f, 0.0f, 1.0f};
  float inv = 1.0f / std::sqrt(n2);
  return {q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

// Converts a 3x3 orthonormal rotation matrix M[row][col] into a unit quaternion.
// Matches Mat3FromQuat() below so Quat(M) -> Mat3(q) is exact.
Quat Mat3ToQuat(const float m[3][3]) {
  float trace = m[0][0] + m[1][1] + m[2][2];
  Quat q;
  if (trace > 0.0f) {
    float s = std::sqrt(trace + 1.0f) * 2.0f;
    q.w = 0.25f * s;
    q.x = (m[2][1] - m[1][2]) / s;
    q.y = (m[0][2] - m[2][0]) / s;
    q.z = (m[1][0] - m[0][1]) / s;
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    float s = std::sqrt(std::max(1.0f + m[0][0] - m[1][1] - m[2][2], 1e-8f)) * 2.0f;
    q.w = (m[2][1] - m[1][2]) / s;
    q.x = 0.25f * s;
    q.y = (m[0][1] + m[1][0]) / s;
    q.z = (m[0][2] + m[2][0]) / s;
  } else if (m[1][1] > m[2][2]) {
    float s = std::sqrt(std::max(1.0f + m[1][1] - m[0][0] - m[2][2], 1e-8f)) * 2.0f;
    q.w = (m[0][2] - m[2][0]) / s;
    q.x = (m[0][1] + m[1][0]) / s;
    q.y = 0.25f * s;
    q.z = (m[1][2] + m[2][1]) / s;
  } else {
    float s = std::sqrt(std::max(1.0f + m[2][2] - m[0][0] - m[1][1], 1e-8f)) * 2.0f;
    q.w = (m[1][0] - m[0][1]) / s;
    q.x = (m[0][2] + m[2][0]) / s;
    q.y = (m[1][2] + m[2][1]) / s;
    q.z = 0.25f * s;
  }
  return NormalizeQuat(q);
}

void QuatToMat3Scaled(const Quat& q, float s, float out[3][3]) {
  float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
  float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
  float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

  out[0][0] = s * (1.0f - 2.0f * (yy + zz));
  out[0][1] = s * (2.0f * (xy - wz));
  out[0][2] = s * (2.0f * (xz + wy));

  out[1][0] = s * (2.0f * (xy + wz));
  out[1][1] = s * (1.0f - 2.0f * (xx + zz));
  out[1][2] = s * (2.0f * (yz - wx));

  out[2][0] = s * (2.0f * (xz - wy));
  out[2][1] = s * (2.0f * (yz + wx));
  out[2][2] = s * (1.0f - 2.0f * (xx + yy));
}

// Shortest-path spherical linear interpolation.
Quat SlerpQuat(Quat a, Quat b, float t) {
  float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  if (dot < 0.0f) {
    b = {-b.x, -b.y, -b.z, -b.w};
    dot = -dot;
  }
  dot = std::clamp(dot, 0.0f, 1.0f);
  if (dot > 0.9995f) {
    return NormalizeQuat({
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t,
    });
  }
  float theta = std::acos(dot);
  float sinTheta = std::sin(theta);
  if (sinTheta <= 1e-6f) {
    return a;
  }
  float w0 = std::sin((1.0f - t) * theta) / sinTheta;
  float w1 = std::sin(t * theta) / sinTheta;
  return NormalizeQuat({
      a.x * w0 + b.x * w1,
      a.y * w0 + b.y * w1,
      a.z * w0 + b.z * w1,
      a.w * w0 + b.w * w1,
  });
}

inline void LoadMat4BE(const uint8_t* srcBE, float out[4][4]) {
  const uint32_t* u = reinterpret_cast<const uint32_t*>(srcBE);
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      uint32_t bits = rex::byte_swap(u[r * 4 + c]);
      float val;
      std::memcpy(&val, &bits, sizeof(val));
      out[r][c] = val;
    }
  }
}

inline void StoreMat4BE(const float in[4][4], uint8_t* dstBE) {
  uint32_t* u = reinterpret_cast<uint32_t*>(dstBE);
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      uint32_t bits;
      std::memcpy(&bits, &in[r][c], sizeof(bits));
      u[r * 4 + c] = rex::byte_swap(bits);
    }
  }
}

// Writes one 4x4 row-major matrix (transposed to 3 vec4 columns: 48B) into
// Vertex Fetch Slot 94 BE guest memory, matching sub_8232EAA0 bit-for-bit.
inline void StoreSlot94MatBE(const float in[4][4], uint8_t* dstBE) {
  uint32_t* u = reinterpret_cast<uint32_t*>(dstBE);
  for (int col = 0; col < 3; col++) {
    for (int row = 0; row < 4; row++) {
      uint32_t bits;
      std::memcpy(&bits, &in[row][col], sizeof(bits));
      u[col * 4 + row] = rex::byte_swap(bits);
    }
  }
}

// Decomposes a big-endian 64B row-major 4x4 affine matrix into rotation quat,
// uniform scale, and translation. Marks validAffine=false for degenerate,
// collapsed (zero-scale dismemberment), non-uniformly scaled, or sheared mats.
DecomposedPose DecomposeFromBE64(const uint8_t* srcBE) {
  DecomposedPose out{};
  LoadMat4BE(srcBE, out.raw);

  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      if (!std::isfinite(out.raw[r][c])) {
        return out;
      }
    }
  }
  if (std::abs(out.raw[0][3]) > 1e-3f || std::abs(out.raw[1][3]) > 1e-3f ||
      std::abs(out.raw[2][3]) > 1e-3f) {
    return out;
  }

  Vec3 r0{out.raw[0][0], out.raw[0][1], out.raw[0][2]};
  Vec3 r1{out.raw[1][0], out.raw[1][1], out.raw[1][2]};
  Vec3 r2{out.raw[2][0], out.raw[2][1], out.raw[2][2]};

  float s0 = Length3(r0);
  float s1 = Length3(r1);
  float s2 = Length3(r2);
  float minS = std::min({s0, s1, s2});
  float maxS = std::max({s0, s1, s2});
  if (minS <= kMinBoneScale || maxS > minS * kMaxScaleAnisotropy) {
    return out;
  }

  Vec3 u0{r0.x / s0, r0.y / s0, r0.z / s0};
  Vec3 u1{r1.x / s1, r1.y / s1, r1.z / s1};
  Vec3 u2{r2.x / s2, r2.y / s2, r2.z / s2};

  if (std::abs(Dot3(u0, u1)) > kMaxAxisDot ||
      std::abs(Dot3(u0, u2)) > kMaxAxisDot ||
      std::abs(Dot3(u1, u2)) > kMaxAxisDot) {
    return out;
  }

  float det = Dot3(u0, Cross3(u1, u2));
  if (det <= kMinDet) {
    return out;
  }

  // Re-orthonormalize via Gram-Schmidt so tiny float drift doesn't skew quat.
  u0 = Normalize3(u0);
  u2 = Normalize3(Cross3(u0, u1));
  u1 = Normalize3(Cross3(u2, u0));

  float rotM[3][3] = {
      {u0.x, u0.y, u0.z},
      {u1.x, u1.y, u1.z},
      {u2.x, u2.y, u2.z},
  };
  out.rot = Mat3ToQuat(rotM);
  out.scale = (s0 + s1 + s2) / 3.0f;
  out.trans = {out.raw[3][0], out.raw[3][1], out.raw[3][2]};
  out.validAffine = true;
  return out;
}

// Synthesizes a row-major 4x4 float matrix between prev and curr at alpha,
// snapping safely to curr on cut flags, invalid/non-uniform decompositions,
// sudden scale pops, or displacements above kTeleportThreshold.
void InterpolateDecomposedMat4(const DecomposedPose& prev,
                               const DecomposedPose& curr, float alpha, bool cut,
                               float out[4][4]) {
  if (cut || alpha >= 0.999f || !prev.validAffine || !curr.validAffine ||
      std::memcmp(prev.raw, curr.raw, sizeof(curr.raw)) == 0) {
    std::memcpy(out, curr.raw, sizeof(curr.raw));
    return;
  }
  if (curr.scale > prev.scale * kMaxScaleJumpRatio ||
      prev.scale > curr.scale * kMaxScaleJumpRatio) {
    std::memcpy(out, curr.raw, sizeof(curr.raw));
    return;
  }
  Vec3 delta{
      curr.trans.x - prev.trans.x,
      curr.trans.y - prev.trans.y,
      curr.trans.z - prev.trans.z,
  };
  if (LengthSq3(delta) > kTeleportThresholdSq) {
    std::memcpy(out, curr.raw, sizeof(curr.raw));
    return;
  }

  Quat q = SlerpQuat(prev.rot, curr.rot, alpha);
  float s = prev.scale + (curr.scale - prev.scale) * alpha;
  float m3[3][3];
  QuatToMat3Scaled(q, s, m3);

  for (int r = 0; r < 3; r++) {
    out[r][0] = m3[r][0];
    out[r][1] = m3[r][1];
    out[r][2] = m3[r][2];
    out[r][3] = curr.raw[r][3];
  }
  out[3][0] = prev.trans.x + delta.x * alpha;
  out[3][1] = prev.trans.y + delta.y * alpha;
  out[3][2] = prev.trans.z + delta.z * alpha;
  out[3][3] = curr.raw[3][3];
}

// --- Guest render table / character slot snapshot ---

struct CharSlotSnap {
  bool valid = false;
  uint16_t health = 0;
  Vec3 pos{};
};

struct RenderEntrySnap {
  uint32_t objectPtr = 0;
  uint32_t modelPtr = 0;
  uint16_t typeId = 0;
  uint8_t kind = 0;
  uint8_t groupSlot = 255;
  uint8_t active = 0;
  int8_t charSlot = -1;
};

struct LogicalPoseKey {
  uint8_t readerKind = 0; // 0 = 980, 1 = AD0, 2 = E00
  uint8_t aux = 0;        // 980: (arg3!=0), AD0: 0, E00: arg3 & 0xFF
  uint16_t partIdx = 0;
  uint32_t entryPtr = 0;

  bool operator==(const LogicalPoseKey& o) const {
    return readerKind == o.readerKind && aux == o.aux && partIdx == o.partIdx &&
           entryPtr == o.entryPtr;
  }
};

struct LogicalPoseKeyHash {
  size_t operator()(const LogicalPoseKey& k) const {
    uint64_t a = (uint64_t(k.entryPtr) << 16) ^ uint64_t(k.partIdx);
    uint64_t b = (uint64_t(k.readerKind) << 8) ^ uint64_t(k.aux);
    return std::hash<uint64_t>{}(a ^ (b * 0x9e3779b97f4a7c15ULL));
  }
};

struct PoseWantedItem {
  LogicalPoseKey key{};
  uint32_t lastArg3 = 0;
};

struct PoseSnap {
  uint32_t guestAddr = 0;
  uint32_t altGuestAddr = 0;
  uint32_t modelPtr = 0;
  uint16_t typeId = 0;
  bool hasRenderedPose = false;
  DecomposedPose pose{};
  DecomposedPose altPose{};
  DecomposedPose renderedPose{};
};

struct FrameSnapshot {
  uint64_t logicFrameId = 0;
  std::array<CharSlotSnap, kCharSlotCount> chars{};
  std::array<RenderEntrySnap, kRenderObjectCount> entries{};
  std::unordered_map<LogicalPoseKey, PoseSnap, LogicalPoseKeyHash> poses;
};

// Cross-thread bridge (main logic thread at sub_822F0CA8 <-> render thread at sub_8247EA10).
std::mutex g_bridgeMutex;
std::vector<PoseWantedItem> g_wantedPoses;
FrameSnapshot g_pendingSnap{};
bool g_pendingReady = false;

// Render-thread owned active snapshots (S[n-1], S[n]) + continuity table.
FrameSnapshot g_prevFrame{};
FrameSnapshot g_currFrame{};
bool g_hasPrevFrame = false;
bool g_hasCurrFrame = false;
std::array<bool, kRenderObjectCount> g_entryCut{};
uint64_t g_activeRenderLogicFrame = 0;

// Wanted poses collected during the current logic frame on the render thread.
std::unordered_map<LogicalPoseKey, uint32_t, LogicalPoseKeyHash> g_renderWantedMap;

// Per-logic-frame pure M_curr cache (written on SF0, reused for all sub-frames
// of the same logic frame).
struct PureCurrEntry {
  uint8_t mat64BE[64] = {};
};
struct PureCurrPart {
  uint32_t partIdx = 0;
  PureCurrEntry entry{};
};
// Flat O(1)-indexed table over kRenderObjectTable entries (retains vector
// capacity across logic frames) with fallback for non-table entries.
std::array<std::vector<PureCurrPart>, kRenderObjectCount> g_pureCurrByIdx{};
std::vector<uint16_t> g_pureCurrActiveIdxs;
std::unordered_map<uint32_t, std::vector<PureCurrPart>> g_pureCurrFallback;

void ClearPureCurrParts() {
  for (uint16_t idx : g_pureCurrActiveIdxs) {
    g_pureCurrByIdx[idx].clear();
  }
  g_pureCurrActiveIdxs.clear();
  g_pureCurrFallback.clear();
}

// Bone palette history per Slot 94 group [0..31].
struct BoneGroupState {
  uint64_t prevLogicFrame = 0;
  uint32_t prevEntryPtr = 0;
  uint32_t prevModelPtr = 0;
  uint16_t prevTypeId = 0;
  uint16_t prevBoneEnd = 0;
  std::array<bool, kMaxBonesPerGroup> prevValid{};
  std::array<DecomposedPose, kMaxBonesPerGroup> prevBones{};

  uint64_t currLogicFrame = 0;
  uint32_t currEntryPtr = 0;
  uint32_t currModelPtr = 0;
  uint16_t currTypeId = 0;
  uint16_t currBoneEnd = 0;
  bool currGroupCut = true;
  std::array<bool, kMaxBonesPerGroup> currValid{};
  std::array<DecomposedPose, kMaxBonesPerGroup> currBones{};
};
std::array<BoneGroupState, kMaxBoneGroups> g_boneGroups{};

thread_local bool t_forceCurrPose = false;
thread_local uint32_t t_activeBoneEntryPtr = 0;

int EntryIndexFromPtr(uint32_t entryPtr) {
  if (entryPtr < kRenderObjectTable) return -1;
  uint32_t off = entryPtr - kRenderObjectTable;
  if (off >= kRenderObjectCount * kRenderObjectStride ||
      (off % kRenderObjectStride) != 0) {
    return -1;
  }
  return int(off / kRenderObjectStride);
}

int CharSlotFromObjectPtr(uint32_t objPtr) {
  if (objPtr < kCharSlotsBase) return -1;
  uint32_t off = objPtr - kCharSlotsBase;
  if (off >= kCharSlotCount * kCharSlotStride ||
      (off % kCharSlotStride) != 0) {
    return -1;
  }
  return int(off / kCharSlotStride);
}

// Resolves a 64B pose matrix guest virtual address matching sub_83751980,
// sub_83751AD0, or sub_83751E00. Returns 0 when the entry's buffer byte is 255
// (identity sentinel) or pointer chain is null.
uint32_t ResolvePoseGuestAddr(uint8_t readerKind, uint32_t entryPtr,
                              uint32_t partIdx, uint32_t arg3,
                              int activeIdxOverride = -1) {
  if (!IsGuestReadable(entryPtr, 24)) return 0;
  uint32_t modelOff = ng2::guest::Load32(entryPtr);

  if (readerKind == 0) {
    uint8_t buf = 255;
    if (arg3 != 0) {
      buf = *ng2::guest::Virtual(entryPtr + 10);
    } else {
      uint32_t activeIdx = (activeIdxOverride >= 0)
                               ? uint32_t(activeIdxOverride)
                               : ng2::guest::Load32(kActiveRenderBufIdx);
      if (activeIdx > 4) return 0;
      buf = *ng2::guest::Virtual(entryPtr + activeIdx + 17);
    }
    if (buf >= 8) return 0;
    uint32_t pool = ng2::guest::Load32(kPoseTablesBase + kPoseTable980Offset +
                                       uint32_t(buf) * 4);
    if (!pool) return 0;
    uint32_t addr = pool + partIdx * 64 + modelOff;
    return IsGuestReadable(addr, 64) ? addr : 0;
  }

  if (readerKind == 1) {
    if (arg3 > 4) return 0;
    uint8_t buf = *ng2::guest::Virtual(entryPtr + arg3 + 17);
    if (buf >= 16) return 0;
    uint32_t pool = ng2::guest::Load32(kPoseTablesBase + kPoseTableAD0Offset +
                                       uint32_t(buf) * 4);
    if (!pool) return 0;
    uint32_t slot = partIdx * 2 + arg3;
    uint32_t addr = pool + slot * 64 + modelOff;
    return IsGuestReadable(addr, 64) ? addr : 0;
  }

  // readerKind == 2 (sub_83751E00)
  if (arg3 > 4) return 0;
  uint8_t buf = *ng2::guest::Virtual(entryPtr + arg3 + 17);
  if (buf >= 16) return 0;
  uint32_t pool = ng2::guest::Load32(kPoseTablesBase + kPoseTableE00Offset +
                                     uint32_t(buf) * 4);
  if (!pool) return 0;
  uint32_t addr = pool + arg3 * 64 + modelOff;
  return IsGuestReadable(addr, 64) ? addr : 0;
}

// Re-evaluates arg3 at sub_822F0CA8 handoff:
// - rk==0 (980) and rk==2 (E00) encode arg3 directly in key.aux.
// - rk==1 (AD0) uses the active render buffer index (0x84C3DA38).
uint32_t ComputeCurrentArg3ForWanted(const PoseWantedItem& item) {
  if (item.key.readerKind == 0 || item.key.readerKind == 2) {
    return item.key.aux;
  }
  uint32_t activeIdx = ng2::guest::Load32(kActiveRenderBufIdx);
  if (activeIdx <= 4) {
    return activeIdx;
  }
  return item.lastArg3;
}

LogicalPoseKey MakePoseKey(uint8_t readerKind, uint32_t entryPtr,
                           uint32_t partIdx, uint32_t arg3) {
  LogicalPoseKey k{};
  k.readerKind = readerKind;
  k.entryPtr = entryPtr;
  k.partIdx = uint16_t(partIdx & 0xFFFFu);
  if (readerKind == 0) {
    k.aux = arg3 != 0 ? 1 : 0;
  } else if (readerKind == 1) {
    k.aux = 0; // AD0 toggles arg3 (0/1) per buffer parity; group same part across frames.
  } else {
    k.aux = uint8_t(arg3 & 0xFFu);
  }
  return k;
}

bool IsEntryCut(uint32_t entryPtr) {
  int idx = EntryIndexFromPtr(entryPtr);
  if (idx >= 0) {
    return g_entryCut[size_t(idx)];
  }
  return !g_hasPrevFrame || !g_hasCurrFrame ||
         g_currFrame.logicFrameId > g_prevFrame.logicFrameId + 1;
}

void RecomputeEntryContinuity() {
  const bool globalCut =
      !g_hasPrevFrame || !g_hasCurrFrame ||
      g_currFrame.logicFrameId > g_prevFrame.logicFrameId + 1;

  for (uint32_t i = 0; i < kRenderObjectCount; i++) {
    if (globalCut) {
      g_entryCut[i] = true;
      continue;
    }
    const RenderEntrySnap& p = g_prevFrame.entries[i];
    const RenderEntrySnap& c = g_currFrame.entries[i];
    if (p.active == 0 || c.active == 0 || p.objectPtr == 0 || c.objectPtr == 0) {
      g_entryCut[i] = true;
      continue;
    }
    if (p.objectPtr != c.objectPtr || p.modelPtr != c.modelPtr ||
        p.typeId != c.typeId || p.kind != c.kind) {
      g_entryCut[i] = true;
      continue;
    }
    if (c.charSlot >= 0) {
      const CharSlotSnap& pc = g_prevFrame.chars[size_t(c.charSlot)];
      const CharSlotSnap& cc = g_currFrame.chars[size_t(c.charSlot)];
      if (!pc.valid || !cc.valid) {
        g_entryCut[i] = true;
        continue;
      }
      Vec3 d{cc.pos.x - pc.pos.x, cc.pos.y - pc.pos.y, cc.pos.z - pc.pos.z};
      if (LengthSq3(d) > kTeleportThresholdSq) {
        g_entryCut[i] = true;
        continue;
      }
    }
    g_entryCut[i] = false;
  }
}

void RotateBoneGroupsForNewFrame() {
  for (uint32_t g = 0; g < kMaxBoneGroups; g++) {
    BoneGroupState& st = g_boneGroups[g];
    const uint16_t copyEnd = std::min<uint16_t>(st.currBoneEnd, kMaxBonesPerGroup);
    if (st.currLogicFrame != 0 &&
        st.currLogicFrame + 1 == g_activeRenderLogicFrame &&
        copyEnd > 0) {
      st.prevLogicFrame = st.currLogicFrame;
      st.prevEntryPtr = st.currEntryPtr;
      st.prevModelPtr = st.currModelPtr;
      st.prevTypeId = st.currTypeId;
      for (uint16_t b = 0; b < copyEnd; b++) {
        st.prevValid[b] = st.currValid[b];
        if (st.currValid[b]) {
          st.prevBones[b] = st.currBones[b];
        }
      }
      if (st.prevBoneEnd > copyEnd) {
        std::fill_n(st.prevValid.begin() + copyEnd, st.prevBoneEnd - copyEnd, false);
      }
      st.prevBoneEnd = copyEnd;
    } else {
      st.prevLogicFrame = 0;
      st.prevEntryPtr = 0;
      st.prevModelPtr = 0;
      st.prevTypeId = 0;
      if (st.prevBoneEnd > 0) {
        std::fill_n(st.prevValid.begin(),
                    std::min<uint16_t>(st.prevBoneEnd, kMaxBonesPerGroup), false);
        st.prevBoneEnd = 0;
      }
    }
    st.currLogicFrame = g_activeRenderLogicFrame;
    st.currEntryPtr = 0;
    st.currModelPtr = 0;
    st.currTypeId = 0;
    st.currGroupCut = true;
    if (copyEnd > 0) {
      std::fill_n(st.currValid.begin(), copyEnd, false);
      st.currBoneEnd = 0;
    }
  }
}

}  // namespace

void Reset() {
  {
    std::lock_guard lock(g_bridgeMutex);
    g_wantedPoses.clear();
    g_pendingSnap = FrameSnapshot{};
    g_pendingReady = false;
  }
  g_readableEpoch.fetch_add(1, std::memory_order_release);
  g_prevFrame = FrameSnapshot{};
  g_currFrame = FrameSnapshot{};
  g_hasPrevFrame = false;
  g_hasCurrFrame = false;
  g_entryCut.fill(true);
  g_activeRenderLogicFrame = 0;
  g_renderWantedMap.clear();
  ClearPureCurrParts();
  for (auto& g : g_boneGroups) {
    g = BoneGroupState{};
  }
}

void OnLogicHandoff() {
  if (!ng2::high_fps::IsActive()) {
    return;
  }
  g_readableEpoch.fetch_add(1, std::memory_order_release);

  FrameSnapshot snap{};
  // LogicFrameId() has not yet been incremented for this render pass; +1 gives
  // the exact id OnRenderSubFrameBegin() assigns when render wakes up.
  snap.logicFrameId = ng2::high_fps::LogicFrameId() + 1;

  for (uint32_t i = 0; i < kCharSlotCount; i++) {
    uint32_t base = kCharSlotsBase + i * kCharSlotStride;
    float px = ng2::guest::LoadFloat(base + kCharPosXOffset + 0);
    float py = ng2::guest::LoadFloat(base + kCharPosXOffset + 4);
    float pz = ng2::guest::LoadFloat(base + kCharPosXOffset + 8);
    uint16_t hp = ng2::guest::Load16(base + kCharHealthOffset);
    CharSlotSnap& cs = snap.chars[i];
    cs.health = hp;
    cs.pos = {px, py, pz};
    cs.valid = std::isfinite(px) && std::isfinite(py) && std::isfinite(pz);
  }

  for (uint32_t i = 0; i < kRenderObjectCount; i++) {
    uint32_t ptr = kRenderObjectTable + i * kRenderObjectStride;
    RenderEntrySnap& re = snap.entries[i];
    re.active = *ng2::guest::Virtual(ptr + 16);
    re.kind = *ng2::guest::Virtual(ptr + 10);
    re.groupSlot = *ng2::guest::Virtual(ptr + 15);
    re.typeId = ng2::guest::Load16(ptr + 4);
    re.objectPtr = ng2::guest::Load32(ptr + 0);
    re.modelPtr = re.objectPtr;
    re.charSlot = int8_t(CharSlotFromObjectPtr(re.objectPtr));
  }

  std::lock_guard lock(g_bridgeMutex);
  snap.poses.reserve(g_wantedPoses.size());
  for (const PoseWantedItem& item : g_wantedPoses) {
    uint32_t arg3 = ComputeCurrentArg3ForWanted(item);
    uint32_t addr = ResolvePoseGuestAddr(item.key.readerKind, item.key.entryPtr,
                                         item.key.partIdx, arg3);
    if (!addr) {
      continue;
    }
    PoseSnap ps{};
    ps.guestAddr = addr;
    ps.modelPtr = ng2::guest::Load32(item.key.entryPtr);
    ps.typeId = ng2::guest::Load16(item.key.entryPtr + 4);
    ps.pose = DecomposeFromBE64(ng2::guest::Virtual(addr));
    // Also capture the opposite double-buffer slot (activeIdx ^ 1) for
    // active-buffer-dependent readers (980 with aux==0, AD0) so a buffer flip
    // between sub_822F0CA8 and sub_8247EA10 still hits the exact snapshot.
    if (item.key.readerKind == 0 && item.key.aux == 0) {
      uint32_t activeIdx = ng2::guest::Load32(kActiveRenderBufIdx);
      if (activeIdx <= 1) {
        uint32_t altAddr = ResolvePoseGuestAddr(
            0, item.key.entryPtr, item.key.partIdx, 0, int(activeIdx ^ 1u));
        if (altAddr && altAddr != addr) {
          ps.altGuestAddr = altAddr;
          ps.altPose = DecomposeFromBE64(ng2::guest::Virtual(altAddr));
        }
      }
    } else if (item.key.readerKind == 1 && arg3 <= 1) {
      uint32_t altAddr = ResolvePoseGuestAddr(
          1, item.key.entryPtr, item.key.partIdx, arg3 ^ 1u);
      if (altAddr && altAddr != addr) {
        ps.altGuestAddr = altAddr;
        ps.altPose = DecomposeFromBE64(ng2::guest::Virtual(altAddr));
      }
    }
    snap.poses.emplace(item.key, ps);
  }

  g_pendingSnap = std::move(snap);
  g_pendingReady = true;
}

void OnRenderFrameBegin() {
  if (!ng2::high_fps::IsActive()) {
    return;
  }

  // Only advance snapshot state once per logic frame (on Sub-Frame 0).
  const uint64_t curLogicId = ng2::high_fps::LogicFrameId();
  if (!ng2::high_fps::IsFirstSubFrame() && curLogicId == g_activeRenderLogicFrame) {
    return;
  }

  g_activeRenderLogicFrame = curLogicId;
  g_readableEpoch.fetch_add(1, std::memory_order_release);
  ClearPureCurrParts();
  RotateBoneGroupsForNewFrame();

  std::vector<PoseWantedItem> nextWanted;
  nextWanted.reserve(g_renderWantedMap.size());
  for (const auto& kv : g_renderWantedMap) {
    nextWanted.push_back({kv.first, kv.second});
  }
  g_renderWantedMap.clear();

  std::lock_guard lock(g_bridgeMutex);
  if (g_pendingReady) {
    if (g_hasCurrFrame) {
      g_prevFrame = std::move(g_currFrame);
      g_hasPrevFrame = true;
    }
    g_currFrame = std::move(g_pendingSnap);
    g_pendingReady = false;
    g_hasCurrFrame = true;
    RecomputeEntryContinuity();
  } else if (g_hasCurrFrame && g_currFrame.logicFrameId != curLogicId) {
    // Defensive cut if a logic frame skipped handoff (e.g. pause / loading).
    g_entryCut.fill(true);
  }
  g_wantedPoses = std::move(nextWanted);
}

bool TryReadInterpolatedPose64(uint8_t readerKind, uint32_t outGuest,
                               uint32_t entryPtr, uint32_t partIdx,
                               uint32_t arg3) {
  if (!ng2::high_fps::IsActive() || !g_hasCurrFrame) {
    return false;
  }

  const bool forceCurr = ForceCurrPose();
  LogicalPoseKey key = MakePoseKey(readerKind, entryPtr, partIdx, arg3);
  if (!forceCurr && ng2::high_fps::IsFirstSubFrame()) {
    g_renderWantedMap[key] = arg3;
  }

  uint32_t liveAddr = ResolvePoseGuestAddr(readerKind, entryPtr, partIdx, arg3);
  if (liveAddr == 0) {
    // Identity sentinel (buf == 255): let original function write identity cleanly.
    return false;
  }

  auto itCurr = g_currFrame.poses.find(key);
  if (itCurr == g_currFrame.poses.end()) {
    return false;
  }
  PoseSnap& currSnap = itCurr->second;
  const DecomposedPose* currPosePtr = nullptr;
  if (currSnap.guestAddr == liveAddr) {
    currPosePtr = &currSnap.pose;
  } else if (currSnap.altGuestAddr == liveAddr) {
    currPosePtr = &currSnap.altPose;
  } else {
    // Newly encountered pose (e.g. spawn or buffer slot swap on this frame):
    // return false so SF0 executes original guest reader and RecordFallbackPose64()
    // freezes live Curr for remaining sub-frames.
    return false;
  }
  if (!forceCurr) {
    currSnap.renderedPose = *currPosePtr;
    currSnap.hasRenderedPose = true;
  }

  const float alpha = ng2::high_fps::SubFrameAlpha();
  if (forceCurr || alpha >= 0.999f) {
    StoreMat4BE(currPosePtr->raw, ng2::guest::Virtual(outGuest));
    return true;
  }

  if (!IsEntryCut(entryPtr) && g_hasPrevFrame) {
    auto itPrev = g_prevFrame.poses.find(key);
    if (itPrev != g_prevFrame.poses.end() &&
        itPrev->second.modelPtr == currSnap.modelPtr &&
        itPrev->second.typeId == currSnap.typeId) {
      const DecomposedPose& prevPose = itPrev->second.hasRenderedPose
                                           ? itPrev->second.renderedPose
                                           : itPrev->second.pose;
      float outMat[4][4];
      InterpolateDecomposedMat4(prevPose, *currPosePtr, alpha, false, outMat);
      StoreMat4BE(outMat, ng2::guest::Virtual(outGuest));
      return true;
    }
  }

  StoreMat4BE(currPosePtr->raw, ng2::guest::Virtual(outGuest));
  return true;
}

void RecordFallbackPose64(uint8_t readerKind, uint32_t entryPtr, uint32_t partIdx,
                          uint32_t arg3, const uint8_t* mat64BE) {
  if (!ng2::high_fps::IsActive() || !g_hasCurrFrame || ForceCurrPose()) {
    return;
  }
  LogicalPoseKey key = MakePoseKey(readerKind, entryPtr, partIdx, arg3);
  if (ng2::high_fps::IsFirstSubFrame()) {
    g_renderWantedMap[key] = arg3;
  }
  uint32_t liveAddr = ResolvePoseGuestAddr(readerKind, entryPtr, partIdx, arg3);
  if (!liveAddr) {
    return;
  }

  PoseSnap ps{};
  ps.guestAddr = liveAddr;
  ps.modelPtr = ng2::guest::Load32(entryPtr);
  ps.typeId = ng2::guest::Load16(entryPtr + 4);
  ps.pose = DecomposeFromBE64(mat64BE);
  ps.renderedPose = ps.pose;
  ps.hasRenderedPose = true;
  g_currFrame.poses[key] = ps;
}

bool ShouldCapturePureCurr(uint32_t entryPtr) {
  if (!ng2::high_fps::IsFirstSubFrame() || ForceCurrPose() || !entryPtr) {
    return false;
  }
  uint8_t groupSlot = *ng2::guest::Virtual(entryPtr + 15);
  return groupSlot < kMaxBoneGroups;
}

const std::vector<PureCurrPart>* FindPureCurrList(uint32_t entryPtr) {
  int idx = EntryIndexFromPtr(entryPtr);
  if (idx >= 0) {
    const auto& vec = g_pureCurrByIdx[size_t(idx)];
    return vec.empty() ? nullptr : &vec;
  }
  auto it = g_pureCurrFallback.find(entryPtr);
  return it != g_pureCurrFallback.end() ? &it->second : nullptr;
}

bool HasPureCurrForFrame(uint32_t entryPtr, uint32_t partIdx) {
  const auto* list = FindPureCurrList(entryPtr);
  if (!list) {
    return false;
  }
  for (const PureCurrPart& p : *list) {
    if (p.partIdx == partIdx) {
      return true;
    }
  }
  return false;
}

void SetForceCurrPose(bool force) { t_forceCurrPose = force; }
bool ForceCurrPose() { return t_forceCurrPose; }

void RecordPureCurrPartMatrix(uint32_t entryPtr, uint32_t partIdx,
                              const uint8_t* mat64BE) {
  if (!entryPtr || !mat64BE) return;
  auto upsert = [&](std::vector<PureCurrPart>& vec) {
    for (PureCurrPart& p : vec) {
      if (p.partIdx == partIdx) {
        std::memcpy(p.entry.mat64BE, mat64BE, 64);
        return;
      }
    }
    PureCurrPart added{};
    added.partIdx = partIdx;
    std::memcpy(added.entry.mat64BE, mat64BE, 64);
    vec.push_back(added);
  };
  int idx = EntryIndexFromPtr(entryPtr);
  if (idx >= 0) {
    auto& vec = g_pureCurrByIdx[size_t(idx)];
    if (vec.empty()) {
      g_pureCurrActiveIdxs.push_back(uint16_t(idx));
    }
    upsert(vec);
    return;
  }
  upsert(g_pureCurrFallback[entryPtr]);
}

BoneBuildScope::BoneBuildScope(uint32_t entryPtr)
    : entryPtr_(entryPtr), prevActiveEntryPtr_(t_activeBoneEntryPtr) {
  t_activeBoneEntryPtr = entryPtr;
  if (ng2::high_fps::SubFrameAlpha() >= 0.999f || !IsGuestReadable(entryPtr, 24)) {
    return;
  }
  const auto* list = FindPureCurrList(entryPtr);
  if (!list) {
    return;
  }
  uint32_t activeIdx = ng2::guest::Load32(kActiveRenderBufIdx);
  if (activeIdx > 4) {
    return;
  }
  uint8_t buf = *ng2::guest::Virtual(entryPtr + activeIdx + 17);
  if (buf >= 8) {
    return;
  }
  uint32_t ptrTableBase =
      ng2::guest::Load32(kPoseTablesBase + kRootCacheTableOffset +
                         uint32_t(buf) * 4);
  if (!ptrTableBase) {
    return;
  }
  uint32_t modelOff = ng2::guest::Load32(entryPtr);

  for (const PureCurrPart& item : *list) {
    if (savedCount_ >= kMaxSavedParts) break;
    uint32_t ptrSlotAddr = ptrTableBase + modelOff + item.partIdx * 4;
    if (!IsGuestReadable(ptrSlotAddr, 4)) continue;
    uint32_t matGuestAddr = ng2::guest::Load32(ptrSlotAddr);
    if (!IsGuestReadable(matGuestAddr, 64)) continue;

    SavedMat& sm = saved_[savedCount_++];
    sm.guestAddr = matGuestAddr;
    uint8_t* hostPtr = ng2::guest::Virtual(matGuestAddr);
    std::memcpy(sm.interpData, hostPtr, 64);
    std::memcpy(hostPtr, item.entry.mat64BE, 64);
  }
}

BoneBuildScope::~BoneBuildScope() {
  for (size_t i = 0; i < savedCount_; i++) {
    std::memcpy(ng2::guest::Virtual(saved_[i].guestAddr),
                saved_[i].interpData, 64);
  }
  t_activeBoneEntryPtr = prevActiveEntryPtr_;
}

bool HandleBonePaletteWrite(uint32_t srcMat4x4, uint32_t groupSlot,
                            uint32_t boneCount, uint32_t startIdx) {
  if (!ng2::high_fps::IsActive()) {
    return false;
  }
  if (boneCount == 0 || boneCount + startIdx > kMaxBonesPerGroup ||
      groupSlot >= kMaxBoneGroups) {
    return false;
  }

  uint32_t dstBase = ng2::guest::Load32(
      kBonePaletteArray + ng2::guest::Load32(kBonePaletteSlotIdx) * 4);
  if (!dstBase) {
    return false;
  }

  BoneGroupState& st = g_boneGroups[groupSlot];
  st.currBoneEnd = std::max<uint16_t>(st.currBoneEnd, uint16_t(startIdx + boneCount));
  const uint32_t entryPtr = t_activeBoneEntryPtr;

  if (ng2::high_fps::IsFirstSubFrame()) {
    if (st.currEntryPtr == 0 && entryPtr != 0) {
      st.currEntryPtr = entryPtr;
      st.currModelPtr = ng2::guest::Load32(entryPtr);
      st.currTypeId = ng2::guest::Load16(entryPtr + 4);

      bool cut = IsEntryCut(entryPtr);
      if (!cut && st.prevLogicFrame + 1 == st.currLogicFrame &&
          st.prevEntryPtr == st.currEntryPtr &&
          st.prevModelPtr == st.currModelPtr &&
          st.prevTypeId == st.currTypeId) {
        st.currGroupCut = false;
      } else {
        st.currGroupCut = true;
      }
    }

    for (uint32_t b = 0; b < boneCount; b++) {
      uint32_t idx = startIdx + b;
      const uint8_t* srcPtr = ng2::guest::Virtual(srcMat4x4 + b * 64);
      st.currBones[idx] = DecomposeFromBE64(srcPtr);
      st.currValid[idx] = true;
    }

    // Check root bone (bone 0) teleport as extra safeguard against camera-cut
    // or spawn warps within a kept slot.
    if (!st.currGroupCut && startIdx == 0 && st.prevValid[0] && st.currValid[0] &&
        st.prevBones[0].validAffine && st.currBones[0].validAffine) {
      Vec3 d{
          st.currBones[0].trans.x - st.prevBones[0].trans.x,
          st.currBones[0].trans.y - st.prevBones[0].trans.y,
          st.currBones[0].trans.z - st.prevBones[0].trans.z,
      };
      if (LengthSq3(d) > kTeleportThresholdSq) {
        st.currGroupCut = true;
      }
    }
  } else if (st.currEntryPtr == 0 && entryPtr != 0) {
    // Bone group became visible after Pass 0 (frustum culled at Pass 0): lock
    // identity and mark cut for this logic frame so currBones capture populates
    // prevBones for the next logic frame.
    st.currEntryPtr = entryPtr;
    st.currModelPtr = ng2::guest::Load32(entryPtr);
    st.currTypeId = ng2::guest::Load16(entryPtr + 4);
    st.currGroupCut = true;
  }

  const float alpha = ng2::high_fps::SubFrameAlpha();
  uint32_t dstGroupAddr = dstBase + groupSlot * kBoneGroupByteStride;

  for (uint32_t b = 0; b < boneCount; b++) {
    uint32_t idx = startIdx + b;
    uint8_t* dstPtr = ng2::guest::Virtual(dstGroupAddr + idx * 48);

    if (!st.currValid[idx]) {
      st.currBones[idx] = DecomposeFromBE64(ng2::guest::Virtual(srcMat4x4 + b * 64));
      st.currValid[idx] = true;
    }
    const DecomposedPose& curr = st.currBones[idx];

    if (alpha >= 0.999f || st.currGroupCut || !st.prevValid[idx]) {
      StoreSlot94MatBE(curr.raw, dstPtr);
      continue;
    }

    float outMat[4][4];
    InterpolateDecomposedMat4(st.prevBones[idx], curr, alpha, false, outMat);
    StoreSlot94MatBE(outMat, dstPtr);
  }

  return true;
}

}  // namespace ng2::high_fps::state_bridge
