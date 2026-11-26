// ng2 - Ninja Gaiden II static recompilation
//
// Phase 3: Deterministic state snapshot bridge & geometry-preserving
// skeletal/pose interpolation for decoupled native 120 FPS rendering.
//
// Captures consistent end-of-logic-frame snapshots S[n] at sub_822F0CA8
// while retaining S[n-1], verifies entity slot / model / displacement continuity,
// and injects shortest-path quaternion SLERP + translation LERP (with safe
// fallback for degenerate / zero-scale / non-uniform-scale matrices) into:
//   - Stage & object 64B pose readers (sub_83751980, sub_83751AD0, sub_83751E00)
//   - Skeletal Vertex Fetch Slot 94 palette writer (sub_8232EAA0)
//   - Pure Curr root tracking across sub_8247DAC8/FFB0 -> sub_82482D18/2F80 so
//     skinned character bones are never double-interpolated.

#pragma once

#include <cstddef>
#include <cstdint>

namespace ng2::high_fps::state_bridge {

// Called on the 60 Hz logic thread inside sub_822F0CA8 right before releasing
// the render thread for the current logic frame. Captures S[n] entity/slot
// metadata and wanted 64-byte stage/object poses into the pending bridge buffer.
void OnLogicHandoff();

// Called on the render thread inside sub_8247EA10 on Sub-Frame 0. Advances
// S[n-1] <- S[n] and activates the pending S[n] snapshot for all sub-frames
// of the current logic frame.
void OnRenderFrameBegin();

// Resets cached per-frame render-thread state (e.g. when returning to title or
// quitting).
void Reset();

// Attempts to satisfy a 64-byte row-major pose read from S[n-1] / S[n] via
// SLERP(R) + LERP(T) at the current sub-frame alpha.
//   readerKind: 0 = sub_83751980, 1 = sub_83751AD0, 2 = sub_83751E00.
// Returns true if outGuest (64 bytes BE) was written cleanly.
bool TryReadInterpolatedPose64(uint8_t readerKind, uint32_t outGuest,
                               uint32_t entryPtr, uint32_t partIdx, uint32_t arg3);

// Registers a fallback pose matrix captured on Sub-Frame 0 (when a pose key
// was not yet in sub_822F0CA8's wanted list) so later sub-frames of the same
// logic frame see a frozen Curr matrix.
void RecordFallbackPose64(uint8_t readerKind, uint32_t entryPtr, uint32_t partIdx,
                          uint32_t arg3, const uint8_t* mat64BE);

// Pure Curr (alpha=1.0) tracking for sub_8247DAC8 / sub_8247FFB0 so that
// sub_82482D18 / sub_82482F80 multiply skeletal hierarchies against un-interpolated
// root matrices before sub_8232EAA0 interpolates the full composite bone matrices.
bool ShouldCapturePureCurr(uint32_t entryPtr);
bool HasPureCurrForFrame(uint32_t entryPtr, uint32_t partIdx);
void SetForceCurrPose(bool force);
bool ForceCurrPose();
void RecordPureCurrPartMatrix(uint32_t entryPtr, uint32_t partIdx,
                              const uint8_t* mat64BE);

// RAII guard for sub_82482D18 / sub_82482F80: temporarily swaps the entry's
// cached 0x83AA56C8+736 part matrices to pure S[n] Curr while bones are built,
// records active entryPtr for sub_8232EAA0, and restores M(alpha) on exit.
class BoneBuildScope {
 public:
  explicit BoneBuildScope(uint32_t entryPtr);
  ~BoneBuildScope();

  BoneBuildScope(const BoneBuildScope&) = delete;
  BoneBuildScope& operator=(const BoneBuildScope&) = delete;

 private:
  struct SavedMat {
    uint32_t guestAddr = 0;
    uint8_t interpData[64] = {};
  };
  static constexpr size_t kMaxSavedParts = 256;
  uint32_t entryPtr_ = 0;
  uint32_t prevActiveEntryPtr_ = 0;
  SavedMat saved_[kMaxSavedParts];
  size_t savedCount_ = 0;
};

// Vertex Fetch Slot 94 skeletal bone palette hook (sub_8232EAA0).
// Returns true when handled directly (wrote SLERPed 3x4 BE transposed bones
// into Slot 94). Returns false to fall back to the original guest function.
bool HandleBonePaletteWrite(uint32_t srcMat4x4, uint32_t groupSlot,
                            uint32_t boneCount, uint32_t startIdx);

}  // namespace ng2::high_fps::state_bridge
