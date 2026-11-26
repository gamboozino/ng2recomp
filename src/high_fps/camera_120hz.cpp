// ng2 - Ninja Gaiden II static recompilation
//
// Native 120 Hz decoupled camera stepping and matrix injection:
// - Records clean 60 Hz logic snapshots Cam[n-1] and Cam[n] around sub_822FA8C8.
// - On render-thread Sub-Frame 0 (alpha=0.5) and Sub-Frame 1 (alpha=1.0):
//   * Free follow-camera mode (Vanilla, Modern Camera, and OT/UT Free Camera):
//     interpolates pivot translation across [Cam[n-1], Cam[n]], applies live
//     half-frame mouse / right-stick input with 120 Hz converted damping
//     k_120 = 1 - sqrt(1 - k_60), online-calibrates sensitivity and axis
//     inversion from observed 60 Hz logic responses, and writes continuous
//     View / Proj / ViewProj + auxiliary matrices into guest camera memory.
//   * Action-camera and cutscene modes: slerps View orientation and lerps
//     Eye / At / FOV between Cam[n-1] and Cam[n], guarded by a camera-cut
//     threshold so shot cuts never smear or interpolate across locations.
// - After both sub-frames finish, restores the saved 60 Hz Cam[n] snapshot
//   before releasing the main loop so 60 Hz gameplay simulation stays pure.

#include "high_fps/camera_120hz.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/types.h>

#include "gameplay.h"
#include "gpu/guest.h"
#include "high_fps/high_fps.h"
#include "high_fps/input_120hz.h"

namespace ng2::high_fps::camera {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;

// Guest CameraManager / Camera layout (verified in docs/native-120fps-plan.md).
constexpr uint32_t kCameraManagerPtrAddr = 0x83B28400;
constexpr uint32_t kProfilePtrAddr = 0x84C559E0;
constexpr uint32_t kCamStride = 5024;

// Offsets inside Camera (cam)
constexpr uint32_t kCamOffEye = 128;
constexpr uint32_t kCamOffAt = 144;
constexpr uint32_t kCamOffAux = 160;
constexpr uint32_t kCamOffFov = 176;
constexpr uint32_t kCamOffRoll = 180;
constexpr uint32_t kCamOffView = 448;
constexpr uint32_t kCamOffProj = 512;
constexpr uint32_t kCamOffViewProj = 576;
constexpr uint32_t kCamOffAuxProj0 = 688;
constexpr uint32_t kCamOffAuxProj1 = 752;
constexpr uint32_t kCamOffSuppress = 992;  // 816 + 176
constexpr uint32_t kCamOffNear = 4700;
constexpr uint32_t kCamOffFar = 4704;
constexpr uint32_t kCamOffMode = 4716;
constexpr uint32_t kCamOffActive = 4764;

// Offsets inside CameraManager (mgr)
constexpr uint32_t kMgrOffView = 0;
constexpr uint32_t kMgrOffProj = 64;
constexpr uint32_t kMgrOffViewProj = 128;
constexpr uint32_t kMgrOffEye = 192;
constexpr uint32_t kMgrOffAt = 208;
constexpr uint32_t kMgrOffAux = 224;
constexpr uint32_t kMgrOffFov = 240;
constexpr uint32_t kMgrOffNear = 268;
constexpr uint32_t kMgrOffFar = 272;
constexpr uint32_t kMgrOffAuxProj0 = 288;
constexpr uint32_t kMgrOffAuxProj1 = 352;
constexpr uint32_t kMgrOffRenderBase = 960;
constexpr uint32_t kMgrOffRenderStride = 320;
constexpr uint32_t kMgrOffParity = 1600;
constexpr uint32_t kMgrOffCamArray = 1604;
constexpr uint32_t kMgrOffActiveIdx = 1612;

struct Vec3 {
  float x = 0.0f, y = 0.0f, z = 0.0f;
};

struct Vec4 {
  float x = 0.0f, y = 0.0f, z = 0.0f, w = 1.0f;
};

struct Mat4 {
  float m[4][4] = {};
};

struct Quat {
  float x = 0.0f, y = 0.0f, z = 0.0f, w = 1.0f;
};

float LoadFloat(uint32_t addr) {
  uint32_t bits = ng2::guest::Load32(addr);
  float v;
  std::memcpy(&v, &bits, 4);
  return v;
}

void StoreFloat(uint32_t addr, float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  *reinterpret_cast<uint32_t*>(ng2::guest::Virtual(addr)) = rex::byte_swap(bits);
}

Vec4 LoadVec4(uint32_t addr) {
  return {LoadFloat(addr), LoadFloat(addr + 4), LoadFloat(addr + 8), LoadFloat(addr + 12)};
}

void StoreVec4(uint32_t addr, const Vec4& v) {
  StoreFloat(addr, v.x);
  StoreFloat(addr + 4, v.y);
  StoreFloat(addr + 8, v.z);
  StoreFloat(addr + 12, v.w);
}

Mat4 LoadMat4(uint32_t addr) {
  Mat4 out;
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      out.m[r][c] = LoadFloat(addr + uint32_t((r * 4 + c) * 4));
    }
  }
  return out;
}

void StoreMat4(uint32_t addr, const Mat4& mat) {
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      StoreFloat(addr + uint32_t((r * 4 + c) * 4), mat.m[r][c]);
    }
  }
}

Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(const Vec3& a, float s) { return {a.x * s, a.y * s, a.z * s}; }

float Dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 Cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float Length(const Vec3& v) { return std::sqrt(Dot(v, v)); }

Vec3 Normalize(const Vec3& v) {
  float len = Length(v);
  if (len <= 1e-6f) {
    return {0.0f, 0.0f, 1.0f};
  }
  return v * (1.0f / len);
}

Vec3 Lerp(const Vec3& a, const Vec3& b, float t) {
  return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
}

float WrapPi(float a) {
  while (a > kPi) a -= kTwoPi;
  while (a < -kPi) a += kTwoPi;
  return a;
}

float LerpAngle(float a, float b, float t) { return WrapPi(a + WrapPi(b - a) * t); }

void FwdToYawPitch(const Vec3& fwd, float* yaw, float* pitch) {
  Vec3 d = Normalize(fwd);
  *yaw = std::atan2(d.x, d.z);
  *pitch = std::asin(std::clamp(d.y, -0.9999f, 0.9999f));
}

Vec3 YawPitchToFwd(float yaw, float pitch) {
  float cp = std::cos(pitch);
  return {std::sin(yaw) * cp, std::sin(pitch), std::cos(yaw) * cp};
}

Mat4 MulMat4(const Mat4& A, const Mat4& B) {
  Mat4 C{};
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      C.m[r][c] = A.m[r][0] * B.m[0][c] + A.m[r][1] * B.m[1][c] + A.m[r][2] * B.m[2][c] +
                  A.m[r][3] * B.m[3][c];
    }
  }
  return C;
}

// Inverts a rigid DX LH row-major view matrix (columns are orthonormal R/U/F,
// row 3 holds (-dot(R,Eye), -dot(U,Eye), -dot(F,Eye), 1)).
Mat4 InvertRigidView(const Mat4& V) {
  Mat4 inv{};
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      inv.m[r][c] = V.m[c][r];
    }
  }
  const float tx = V.m[3][0], ty = V.m[3][1], tz = V.m[3][2];
  inv.m[3][0] = -(tx * inv.m[0][0] + ty * inv.m[1][0] + tz * inv.m[2][0]);
  inv.m[3][1] = -(tx * inv.m[0][1] + ty * inv.m[1][1] + tz * inv.m[2][1]);
  inv.m[3][2] = -(tx * inv.m[0][2] + ty * inv.m[1][2] + tz * inv.m[2][2]);
  inv.m[3][3] = 1.0f;
  return inv;
}

// Rebuilds NG2 row-major View matrix matching sub_822F9A10 layout exactly:
// col 0 = right = Normalize(Cross(fwd, worldUp)),
// col 1 = up    = Cross(right, fwd),
// col 2 = +fwd  (camera looks along +Z in view space; Proj.m[2][3] = +1, det = -1),
// post-multiplied by Rz(roll) from sub_822F6570 when |roll| > 1e-6.
Mat4 BuildLookAtNG2(const Vec3& eye, const Vec3& at, float roll = 0.0f) {
  Vec3 fwd = Normalize(at - eye);
  Vec3 worldUp = std::abs(fwd.y) > 0.9999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{0.0f, 1.0f, 0.0f};
  Vec3 right = Normalize(Cross(fwd, worldUp));
  Vec3 up = Cross(right, fwd);
  Vec3 zCol = fwd;

  Mat4 v{};
  v.m[0][0] = right.x;
  v.m[0][1] = up.x;
  v.m[0][2] = zCol.x;
  v.m[1][0] = right.y;
  v.m[1][1] = up.y;
  v.m[1][2] = zCol.y;
  v.m[2][0] = right.z;
  v.m[2][1] = up.z;
  v.m[2][2] = zCol.z;
  v.m[3][0] = -Dot(right, eye);
  v.m[3][1] = -Dot(up, eye);
  v.m[3][2] = -Dot(zCol, eye);
  v.m[3][3] = 1.0f;

  if (std::abs(roll) > 1e-6f) {
    const float c = std::cos(roll);
    const float s = std::sin(roll);
    Mat4 rz{};
    rz.m[0][0] = c;
    rz.m[0][1] = s;
    rz.m[1][0] = -s;
    rz.m[1][1] = c;
    rz.m[2][2] = 1.0f;
    rz.m[3][3] = 1.0f;
    v = MulMat4(v, rz);
  }
  return v;
}

Quat Mat3ToQuat(const Mat4& m) {
  // Extracts rotation from the upper 3x3 basis columns (R, U, F).
  const float m00 = m.m[0][0], m11 = m.m[1][1], m22 = m.m[2][2];
  const float trace = m00 + m11 + m22;
  Quat q{};
  if (trace > 0.0f) {
    float s = std::sqrt(trace + 1.0f) * 2.0f;
    q.w = 0.25f * s;
    q.x = (m.m[1][2] - m.m[2][1]) / s;
    q.y = (m.m[2][0] - m.m[0][2]) / s;
    q.z = (m.m[0][1] - m.m[1][0]) / s;
  } else if (m00 > m11 && m00 > m22) {
    float s = std::sqrt(std::max(1e-6f, 1.0f + m00 - m11 - m22)) * 2.0f;
    q.w = (m.m[1][2] - m.m[2][1]) / s;
    q.x = 0.25f * s;
    q.y = (m.m[0][1] + m.m[1][0]) / s;
    q.z = (m.m[2][0] + m.m[0][2]) / s;
  } else if (m11 > m22) {
    float s = std::sqrt(std::max(1e-6f, 1.0f + m11 - m00 - m22)) * 2.0f;
    q.w = (m.m[2][0] - m.m[0][2]) / s;
    q.x = (m.m[0][1] + m.m[1][0]) / s;
    q.y = 0.25f * s;
    q.z = (m.m[1][2] + m.m[2][1]) / s;
  } else {
    float s = std::sqrt(std::max(1e-6f, 1.0f + m22 - m00 - m11)) * 2.0f;
    q.w = (m.m[0][1] - m.m[1][0]) / s;
    q.x = (m.m[2][0] + m.m[0][2]) / s;
    q.y = (m.m[1][2] + m.m[2][1]) / s;
    q.z = 0.25f * s;
  }
  float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (len > 1e-6f) {
    q.x /= len;
    q.y /= len;
    q.z /= len;
    q.w /= len;
  } else {
    q = {0.0f, 0.0f, 0.0f, 1.0f};
  }
  return q;
}

Quat ConjQuat(const Quat& q) { return {-q.x, -q.y, -q.z, q.w}; }

Quat MulQuat(const Quat& a, const Quat& b) {
  return {
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}

// Transposes NG2's View matrix (where Col 2 = +Fwd, det = -1) into a proper
// (det = +1) CamToWorld rotation quaternion by negating Row 2.
Quat ViewToCamQuat(const Mat4& view) {
  Mat4 camToWorld{};
  for (int c = 0; c < 3; c++) {
    camToWorld.m[0][c] = view.m[c][0];
    camToWorld.m[1][c] = view.m[c][1];
    camToWorld.m[2][c] = -view.m[c][2];
  }
  camToWorld.m[3][3] = 1.0f;
  return Mat3ToQuat(camToWorld);
}

// Updates a 64-byte XNA::Frustum (stored at Camera +688/+752 and CameraMgr render
// slots +176/+240: Origin vec4, Orientation quat, 4 slopes vec4, Near/Far vec4).
Mat4 UpdateFrustumForSubFrame(const Mat4& baseFrustum, const Mat4& baseView, const Vec3& eye,
                              float baseFov, float fov, const Mat4& view) {
  Mat4 out = baseFrustum;
  // +0: Origin (vec3 eye, preserve .w)
  out.m[0][0] = eye.x;
  out.m[0][1] = eye.y;
  out.m[0][2] = eye.z;

  // +16: Orientation quaternion. sub_837F1FB0 builds qBase = qOffset * qV0 in
  // row-vector convention (v' = v * R_row(q)), where R_row(q1) * R_row(q2) = R_row(q2 * q1).
  // Therefore qNew = qBase * conj(qV0) * qV1 preserves the exact sub_837F1FB0 offset.
  const Quat qBase{baseFrustum.m[1][0], baseFrustum.m[1][1], baseFrustum.m[1][2],
                   baseFrustum.m[1][3]};
  const float qLen2 = qBase.x * qBase.x + qBase.y * qBase.y + qBase.z * qBase.z + qBase.w * qBase.w;
  if (qLen2 > 1e-4f) {
    const Quat qV0 = ViewToCamQuat(baseView);
    const Quat qV1 = ViewToCamQuat(view);
    const Quat qNew = MulQuat(qBase, MulQuat(ConjQuat(qV0), qV1));
    const float len =
        std::sqrt(qNew.x * qNew.x + qNew.y * qNew.y + qNew.z * qNew.z + qNew.w * qNew.w);
    if (len > 1e-6f) {
      out.m[1][0] = qNew.x / len;
      out.m[1][1] = qNew.y / len;
      out.m[1][2] = qNew.z / len;
      out.m[1][3] = qNew.w / len;
    }
  }

  // +32: RightSlope, LeftSlope, TopSlope, BottomSlope
  if (std::abs(fov - baseFov) > 1e-4f && fov > 0.05f && baseFov > 0.05f) {
    const float ratio = std::tan(fov * 0.5f) / std::tan(baseFov * 0.5f);
    for (int c = 0; c < 4; c++) {
      out.m[2][c] *= ratio;
    }
  }
  return out;
}

Quat SlerpQuat(Quat a, Quat b, float t) {
  float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  if (dot < 0.0f) {
    b = {-b.x, -b.y, -b.z, -b.w};
    dot = -dot;
  }
  if (dot > 0.9995f) {
    Quat q{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t,
           a.w + (b.w - a.w) * t};
    float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return len > 1e-6f ? Quat{q.x / len, q.y / len, q.z / len, q.w / len} : a;
  }
  float theta0 = std::acos(std::clamp(dot, -1.0f, 1.0f));
  float theta = theta0 * t;
  float sin0 = std::sin(theta0);
  float s0 = std::cos(theta) - dot * std::sin(theta) / sin0;
  float s1 = std::sin(theta) / sin0;
  return {a.x * s0 + b.x * s1, a.y * s0 + b.y * s1, a.z * s0 + b.z * s1, a.w * s0 + b.w * s1};
}

// Rebuilds NG2 View matrix from CamToWorld quaternion (from ViewToCamQuat) and eye.
Mat4 BuildViewFromCamQuatAndEye(const Quat& q, const Vec3& eye) {
  const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
  const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
  const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

  // CamToWorld rows (v' = v * R_row(q)): transpose into View columns, negating col 2.
  Vec3 right{1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy)};
  Vec3 up{2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx)};
  Vec3 zCol{-(2.0f * (xz + wy)), -(2.0f * (yz - wx)), -(1.0f - 2.0f * (xx + yy))};

  Mat4 v{};
  v.m[0][0] = right.x;
  v.m[0][1] = up.x;
  v.m[0][2] = zCol.x;
  v.m[1][0] = right.y;
  v.m[1][1] = up.y;
  v.m[1][2] = zCol.y;
  v.m[2][0] = right.z;
  v.m[2][1] = up.z;
  v.m[2][2] = zCol.z;
  v.m[3][0] = -Dot(right, eye);
  v.m[3][1] = -Dot(up, eye);
  v.m[3][2] = -Dot(zCol, eye);
  v.m[3][3] = 1.0f;
  return v;
}

Vec4 ProjectWorldPoint(const Mat4& vp, const Vec3& p) {
  return {
      p.x * vp.m[0][0] + p.y * vp.m[1][0] + p.z * vp.m[2][0] + vp.m[3][0],
      p.x * vp.m[0][1] + p.y * vp.m[1][1] + p.z * vp.m[2][1] + vp.m[3][1],
      p.x * vp.m[0][2] + p.y * vp.m[1][2] + p.z * vp.m[2][2] + vp.m[3][2],
      p.x * vp.m[0][3] + p.y * vp.m[1][3] + p.z * vp.m[2][3] + vp.m[3][3],
  };
}

struct CameraSnapshot {
  bool valid = false;
  uint64_t logicFrameId = 0;
  uint32_t mgr = 0;
  uint32_t cam = 0;
  uint32_t mode = 0;
  bool followRan = false;
  bool actionRan = false;
  bool modernCam = false;
  bool otUtFree = false;
  bool suppressed = false;
  bool inCutscene = false;

  Vec4 eye{};
  Vec4 at{};
  Vec4 aux{};
  float fov = 0.785398f;
  float roll = 0.0f;
  float dist = 3.5f;
  float yaw = 0.0f;
  float pitch = 0.0f;

  Mat4 view{};
  Mat4 proj{};
  Mat4 viewProj{};
  Mat4 auxProj0{};
  Mat4 auxProj1{};
  input::LogicInputSample logicInput{};
};

std::mutex g_mutex;

// Flags recorded during sub_822FA8C8 on the logic thread.
bool g_tickFollowRan = false;
bool g_tickActionRan = false;
bool g_tickModern = false;
bool g_tickOtUtFree = false;
bool g_tickSuppressed = false;

// Metadata captured during sub_822FA8C8 on the 60 Hz logic thread and committed
// atomically at sub_822F0CA8 (OnLogicHandoff), before main-loop frame n+1 starts.
struct CameraMeta {
  bool valid = false;
  uint64_t logicFrameId = 0;
  uint32_t mgr = 0;
  uint32_t cam = 0;
  uint32_t mode = 0;
  bool followRan = false;
  bool actionRan = false;
  bool modernCam = false;
  bool otUtFree = false;
  bool suppressed = false;
  bool inCutscene = false;
  float roll = 0.0f;
  input::LogicInputSample logicInput{};
};

CameraMeta g_liveMeta{};
bool g_hasLiveMeta = false;
CameraMeta g_handoffMeta{};
bool g_hasHandoff = false;

// 60 Hz logic double-buffer latched only at Sub-Frame 0 boundary from the
// render thread's exclusive parity slot + committed handoff metadata.
CameraSnapshot g_prev{};
CameraSnapshot g_curr{};

// Active 120 Hz rendered frame matrices (for RestoreLogicCamera & 2D HUD sync).
bool g_injectedActive = false;
Mat4 g_subFrameView{};
Mat4 g_subFrameProj{};
Mat4 g_subFrameViewProj{};
uint32_t g_injectedRenderSlot = 0;

// Continuous 120 Hz follow-camera filter & sensitivity state.
uint64_t g_lastHandledLogicId = 0;
float g_visYaw = 0.0f;
float g_visPitch = 0.0f;
float g_velYaw = 0.0f;
float g_velPitch = 0.0f;
float g_residualYaw = 0.0f;
float g_residualPitch = 0.0f;
float g_signedGainYaw = -0.036f;   // signed rad per normalized 60 Hz full-throw input
float g_signedGainPitch = 0.025f;
bool g_visInit = false;
bool g_gainCalibratedYaw = false;
bool g_gainCalibratedPitch = false;
uint32_t g_oppositeSignStreakYaw = 0;
uint32_t g_oppositeSignStreakPitch = 0;
bool g_lastInvX = false;
bool g_lastInvY = false;
bool g_lastInvInit = false;

void ReadProfileCameraTuning(bool* invX, bool* invY, float* speedNorm) {
  const uint32_t profile = ng2::guest::Load32(kProfilePtrAddr);
  if (!profile) {
    *invX = false;
    *invY = false;
    *speedNorm = 0.5f;
    return;
  }
  const uint32_t inv = *ng2::guest::Virtual(profile + 60);
  const uint32_t speed = *ng2::guest::Virtual(profile + 63);
  *invX = (inv == 1 || inv == 3);
  *invY = (inv == 2 || inv == 3);
  *speedNorm = std::clamp(float(speed) * 0.01f, 0.0f, 1.0f);
}

// Normalizes raw stick / mouse axis into [-1, 1] (or un-clamped for mouse calibration).
float ShapeAxisInput(int32_t raw, bool isMouse, bool clampMouse = true) {
  if (isMouse) {
    float v = float(raw) / 32767.0f;
    return clampMouse ? std::clamp(v, -1.0f, 1.0f) : v;
  }
  float v = std::clamp(float(raw) / 32767.0f, -1.0f, 1.0f);
  constexpr float kDeadzone = 0.22f;
  float av = std::abs(v);
  if (av <= kDeadzone) {
    return 0.0f;
  }
  float n = (av - kDeadzone) / (1.0f - kDeadzone);
  // Mirrors NG2's smooth cubic+quartic curve without extreme low-end sluggishness.
  float shaped = n * n * (0.5f + 0.5f * n);
  return v >= 0.0f ? shaped : -shaped;
}

bool IsCameraCut(const CameraSnapshot& a, const CameraSnapshot& b) {
  if (!a.valid || !b.valid || a.cam != b.cam) {
    return true;
  }
  if (b.logicFrameId > a.logicFrameId + 1) {
    return true;
  }
  Vec3 ea{a.eye.x, a.eye.y, a.eye.z}, eb{b.eye.x, b.eye.y, b.eye.z};
  Vec3 aa{a.at.x, a.at.y, a.at.z}, ab{b.at.x, b.at.y, b.at.z};
  const float scaleDist = std::max({a.dist, b.dist, 1.0f});
  const float cutDist = std::max(scaleDist * 0.85f, 3.0f);
  if (Length(eb - ea) > cutDist || Length(ab - aa) > cutDist) {
    return true;
  }
  Vec3 fa = Normalize(aa - ea), fb = Normalize(ab - eb);
  if (Dot(fa, fb) < 0.45f) {
    return true;
  }
  if (std::abs(b.fov - a.fov) > 0.25f) {
    return true;
  }
  return false;
}

bool IsFreeFollowEligible(const CameraSnapshot& s) {
  if (!s.valid || s.inCutscene || s.suppressed) {
    return false;
  }
  if (s.actionRan && !s.otUtFree) {
    return false;
  }
  if (s.followRan && (s.mode == 1 || s.mode == 2 || s.mode == 3 || s.mode == 6 || s.otUtFree)) {
    return true;
  }
  return false;
}

// Writes interpolated/stepped View, Proj, ViewProj and Eye/At/FOV to Camera,
// CameraManager and the active render-thread double-buffer parity slot.
void WriteSubFrameToGuest(const CameraSnapshot& base, const Vec3& eye, const Vec3& at, float fov,
                          const Mat4& view, const Mat4& proj) {
  const uint32_t cam = base.cam;
  const uint32_t mgr = base.mgr;
  if (!cam || !mgr) {
    return;
  }

  const Mat4 viewProj = MulMat4(view, proj);
  // Camera +688/+752 (and render slots +176/+240) store 64-byte XNA::Frustum
  // structs used by sub_8247A518 -> sub_82364CD0 -> sub_837F11C8 stage culling.
  const Mat4 aux0 = UpdateFrustumForSubFrame(base.auxProj0, base.view, eye, base.fov, fov, view);
  const Mat4 aux1 = UpdateFrustumForSubFrame(base.auxProj1, base.view, eye, base.fov, fov, view);

  const Vec4 eye4{eye.x, eye.y, eye.z, base.eye.w};
  const Vec4 at4{at.x, at.y, at.z, base.at.w};

  // Update ONLY the render-thread double-buffer parity slot ((parity ^ 1) in
  // sub_822F16D0). Leaving Camera (cam) and CameraManager +0..+352 untouched
  // keeps the concurrent 60 Hz main loop completely isolated and race-free.
  const uint32_t renderSlot = (ng2::guest::Load32(mgr + kMgrOffParity) ^ 1u) & 1u;
  const uint32_t slotBase = mgr + kMgrOffRenderBase + renderSlot * kMgrOffRenderStride;
  StoreMat4(slotBase + 0, view);
  StoreMat4(slotBase + 64, proj);
  StoreVec4(slotBase + 128, eye4);
  StoreVec4(slotBase + 144, at4);
  StoreMat4(slotBase + 176, aux0);
  StoreMat4(slotBase + 240, aux1);
  StoreFloat(slotBase + 304, fov);

  g_injectedActive = true;
  g_injectedRenderSlot = renderSlot;
  g_subFrameView = view;
  g_subFrameProj = proj;
  g_subFrameViewProj = viewProj;

  static const bool kCamLog = [] {
    const char* v = std::getenv("NG2_CAMERA_120HZ_LOG");
    return v && v[0] == '1';
  }();
  if (kCamLog) {
    static uint32_t loggedCount = 0;
    if (loggedCount++ % 2 == 0) {
      float ry = 0.0f, rp = 0.0f;
      FwdToYawPitch(at - eye, &ry, &rp);
      const input::CameraInputSample subIn = input::LatestCameraInput();
      REXLOG_INFO("cam120: frame={} {} mode={} free={} cut={} yaw={:.5f} pitch={:.5f} "
                  "eye=({:.2f},{:.2f},{:.2f}) vel=({:.5f},{:.5f}) rx={} ry={} mouse={}",
                  base.logicFrameId, ng2::high_fps::SubFrameIndex(), base.mode,
                  IsFreeFollowEligible(base) ? 1 : 0, IsCameraCut(g_prev, base) ? 1 : 0, ry, rp,
                  eye.x, eye.y, eye.z, g_velYaw, g_velPitch, subIn.rx, subIn.ry,
                  subIn.isMouse ? 1 : 0);
    }
  }
}

uint32_t ActiveCamAddr(uint32_t mgr) {
  if (!mgr) return 0;
  uint32_t idx = ng2::guest::Load32(mgr + kMgrOffActiveIdx);
  uint32_t arr = ng2::guest::Load32(mgr + kMgrOffCamArray);
  if (!arr || idx >= 8) return 0;
  return arr + idx * kCamStride;
}

}  // namespace

void OnFollowCameraTick(uint32_t camPtr, bool modern, bool suppressedOrLocked) {
  const uint32_t mgr = ng2::guest::Load32(kCameraManagerPtrAddr);
  if (camPtr != ActiveCamAddr(mgr)) {
    return;
  }
  g_tickFollowRan = true;
  g_tickModern = modern;
  g_tickSuppressed = suppressedOrLocked || (ng2::guest::Load32(camPtr + kCamOffSuppress) != 0);
}

void OnActionCameraTick(uint32_t camPtr, bool otUtFreeCam) {
  const uint32_t mgr = ng2::guest::Load32(kCameraManagerPtrAddr);
  if (camPtr != ActiveCamAddr(mgr)) {
    return;
  }
  g_tickActionRan = true;
  g_tickOtUtFree = otUtFreeCam;
}

void OnCameraUpdateEnd(uint32_t camPtr) {
  const uint32_t mgr = ng2::guest::Load32(kCameraManagerPtrAddr);
  const uint32_t activeCam = ActiveCamAddr(mgr);
  if (!mgr || !activeCam || camPtr != activeCam) {
    return;
  }
  if (ng2::guest::Load32(camPtr + kCamOffActive) == 0) {
    return;
  }

  CameraMeta meta{};
  meta.valid = true;
  meta.mgr = mgr;
  meta.cam = camPtr;
  meta.mode = ng2::guest::Load32(camPtr + kCamOffMode);
  meta.followRan = g_tickFollowRan;
  meta.actionRan = g_tickActionRan;
  meta.modernCam = g_tickModern;
  meta.otUtFree = g_tickOtUtFree;
  meta.suppressed = g_tickSuppressed || (ng2::guest::Load32(camPtr + kCamOffSuppress) != 0);
  meta.inCutscene = ng2::gameplay::InCutscene();
  meta.roll = LoadFloat(camPtr + kCamOffRoll);

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_liveMeta = meta;
    g_hasLiveMeta = true;
  }

  // Reset per-tick detection flags for the next camera update.
  g_tickFollowRan = false;
  g_tickActionRan = false;
  g_tickOtUtFree = false;
  g_tickSuppressed = false;
}

void OnLogicHandoff() {
  if (!ng2::high_fps::IsActive()) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_hasLiveMeta) {
    g_handoffMeta = g_liveMeta;
    // LogicFrameId() has not yet been incremented for this render pass; +1 matches
    // the exact id OnRenderSubFrameBegin() assigns when the render thread wakes up.
    g_handoffMeta.logicFrameId = ng2::high_fps::LogicFrameId() + 1;
    g_handoffMeta.logicInput = input::LatestLogicInput();
    g_hasHandoff = true;
    g_hasLiveMeta = false;
  } else {
    g_handoffMeta = CameraMeta{};
    g_hasHandoff = false;
  }
}

void ApplySubFrameCamera(bool isFirstPass, float alpha, float dtSec) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_injectedActive = false;

  if (!ng2::high_fps::IsActive()) {
    return;
  }

  // On Pass 0 entry of a logic frame, latch the authoritative render-ready camera slot
  // ((parity ^ 1) in sub_822F16D0, exclusively owned by the render thread across
  // all sub-frames of this logic frame) paired with the handoff metadata from
  // sub_822F0CA8. This is completely race-free against concurrent main-loop frame n+1.
  if (isFirstPass) {
    if (g_hasHandoff && g_handoffMeta.valid && g_handoffMeta.mgr) {
      const uint32_t mgr = g_handoffMeta.mgr;
      const uint32_t renderSlot = (ng2::guest::Load32(mgr + kMgrOffParity) ^ 1u) & 1u;
      const uint32_t slotBase = mgr + kMgrOffRenderBase + renderSlot * kMgrOffRenderStride;

      const Mat4 slotView = LoadMat4(slotBase + 0);
      const Mat4 slotProj = LoadMat4(slotBase + 64);
      const bool slotValid = std::isfinite(slotView.m[3][3]) &&
                             std::abs(slotView.m[3][3] - 1.0f) < 0.05f &&
                             std::isfinite(slotProj.m[2][3]) &&
                             std::abs(slotProj.m[2][3]) > 0.1f;

      if (slotValid) {
        CameraSnapshot snap{};
        snap.valid = true;
        snap.logicFrameId = g_handoffMeta.logicFrameId;
        snap.mgr = mgr;
        snap.cam = g_handoffMeta.cam;
        snap.mode = g_handoffMeta.mode;
        snap.followRan = g_handoffMeta.followRan;
        snap.actionRan = g_handoffMeta.actionRan;
        snap.modernCam = g_handoffMeta.modernCam;
        snap.otUtFree = g_handoffMeta.otUtFree;
        snap.suppressed = g_handoffMeta.suppressed;
        snap.inCutscene = g_handoffMeta.inCutscene;
        snap.roll = g_handoffMeta.roll;

        snap.view = slotView;
        snap.proj = slotProj;
        snap.viewProj = MulMat4(slotView, slotProj);
        snap.eye = LoadVec4(slotBase + 128);
        snap.at = LoadVec4(slotBase + 144);
        snap.aux = LoadVec4(slotBase + 160);
        snap.auxProj0 = LoadMat4(slotBase + 176);
        snap.auxProj1 = LoadMat4(slotBase + 240);
        snap.fov = LoadFloat(slotBase + 304);
        snap.logicInput = g_handoffMeta.logicInput;

        const Vec3 e{snap.eye.x, snap.eye.y, snap.eye.z};
        const Vec3 a{snap.at.x, snap.at.y, snap.at.z};
        snap.dist = std::max(0.05f, Length(a - e));
        FwdToYawPitch(a - e, &snap.yaw, &snap.pitch);

        if (g_curr.valid) {
          g_prev = g_curr;
        } else {
          g_prev = snap;
        }
        g_curr = snap;
      }
      g_hasHandoff = false;
    } else {
      // Camera did not update this logic frame (menu, pause, loading): invalidate
      // so we don't overwrite the slot with stale state.
      g_curr.valid = false;
    }
  }

  if (!g_curr.valid || !g_curr.mgr) {
    g_velYaw = 0.0f;
    g_velPitch = 0.0f;
    g_residualYaw = 0.0f;
    g_residualPitch = 0.0f;
    g_visInit = false;
    return;
  }

  const uint64_t curLogicId = ng2::high_fps::LogicFrameId();
  if (curLogicId > g_curr.logicFrameId + 2) {
    // Stale snapshot (game paused, menu, or loading screen skipped camera update).
    g_velYaw = 0.0f;
    g_velPitch = 0.0f;
    g_residualYaw = 0.0f;
    g_residualPitch = 0.0f;
    g_visInit = false;
    return;
  }

  const bool cut = IsCameraCut(g_prev, g_curr);
  const CameraSnapshot& prev = cut ? g_curr : g_prev;
  const CameraSnapshot& curr = g_curr;

  if (IsFreeFollowEligible(curr)) {
    const input::CameraInputSample subIn = input::LatestCameraInput();
    const input::LogicInputSample& logIn = curr.logicInput;

    bool invX = false, invY = false;
    float speedNorm = 0.5f;
    ReadProfileCameraTuning(&invX, &invY, &speedNorm);

    const float signYaw = invX ? 1.0f : -1.0f;
    const float signPitch = invY ? -1.0f : 1.0f;
    const float speedScale = 0.65f + 0.70f * speedNorm;
    if (!g_lastInvInit || invX != g_lastInvX || invY != g_lastInvY) {
      g_lastInvInit = true;
      g_lastInvX = invX;
      g_lastInvY = invY;
      g_gainCalibratedYaw = false;
      g_gainCalibratedPitch = false;
      g_oppositeSignStreakYaw = 0;
      g_oppositeSignStreakPitch = 0;
    }
    if (!g_gainCalibratedYaw) {
      g_signedGainYaw = 0.036f * speedScale * signYaw;
    }
    if (!g_gainCalibratedPitch) {
      g_signedGainPitch = 0.025f * speedScale * signPitch;
    }

    // Calibrate online sensitivity once on each new 60 Hz logic frame.
    const bool newLogicFrame = (curr.logicFrameId != g_lastHandledLogicId);
    if (newLogicFrame) {
      g_lastHandledLogicId = curr.logicFrameId;

      if (!cut && prev.valid && IsFreeFollowEligible(prev)) {
        const float lx = ShapeAxisInput(logIn.isMouse ? logIn.unclampedRx : logIn.rx,
                                        logIn.isMouse, false);
        const float ly = ShapeAxisInput(logIn.isMouse ? logIn.unclampedRy : logIn.ry,
                                        logIn.isMouse, false);
        const float dCamYaw = WrapPi(curr.yaw - prev.yaw);
        const float dCamPitch = curr.pitch - prev.pitch;

        // Require meaningful stick/mouse throw and reasonable camera delta; never
        // flip polarity on a single frame of combat auto-camera / RT-center noise.
        if (std::abs(lx) > 0.35f && std::abs(dCamYaw) > 0.004f && std::abs(dCamYaw) < 0.25f) {
          const float rawObs = dCamYaw / lx;
          const float clampedMag = std::clamp(std::abs(rawObs), 0.012f, 0.12f);
          const float obs = (rawObs >= 0.0f) ? clampedMag : -clampedMag;
          if (obs * g_signedGainYaw > 0.0f) {
            g_oppositeSignStreakYaw = 0;
            g_signedGainYaw = g_signedGainYaw * 0.80f + obs * 0.20f;
            g_gainCalibratedYaw = true;
          } else if (++g_oppositeSignStreakYaw >= 3) {
            g_oppositeSignStreakYaw = 0;
            g_signedGainYaw = obs;
            g_gainCalibratedYaw = true;
          }
        } else {
          g_oppositeSignStreakYaw = 0;
        }

        if (std::abs(ly) > 0.35f && std::abs(dCamPitch) > 0.004f && std::abs(dCamPitch) < 0.18f) {
          const float rawObs = dCamPitch / ly;
          const float clampedMag = std::clamp(std::abs(rawObs), 0.010f, 0.09f);
          const float obs = (rawObs >= 0.0f) ? clampedMag : -clampedMag;
          if (obs * g_signedGainPitch > 0.0f) {
            g_oppositeSignStreakPitch = 0;
            g_signedGainPitch = g_signedGainPitch * 0.80f + obs * 0.20f;
            g_gainCalibratedPitch = true;
          } else if (++g_oppositeSignStreakPitch >= 3) {
            g_oppositeSignStreakPitch = 0;
            g_signedGainPitch = obs;
            g_gainCalibratedPitch = true;
          }
        } else {
          g_oppositeSignStreakPitch = 0;
        }

        // If the game blocked or clamped camera rotation (wall collision, combat
        // hit-stop, or RT center-camera), damp predicted velocity immediately.
        const float expectedYaw = lx * g_signedGainYaw;
        if (std::abs(expectedYaw) > 0.005f &&
            (dCamYaw * expectedYaw < 0.0f || std::abs(dCamYaw) < 0.15f * std::abs(expectedYaw))) {
          g_velYaw *= 0.35f;
        }
        const float expectedPitch = ly * g_signedGainPitch;
        if (std::abs(expectedPitch) > 0.005f &&
            (dCamPitch * expectedPitch < 0.0f ||
             std::abs(dCamPitch) < 0.15f * std::abs(expectedPitch))) {
          g_velPitch *= 0.35f;
        }
      }

      if (!g_visInit || cut) {
        g_visYaw = curr.yaw;
        g_visPitch = curr.pitch;
        g_velYaw = 0.0f;
        g_velPitch = 0.0f;
        g_residualYaw = 0.0f;
        g_residualPitch = 0.0f;
        g_visInit = true;
      } else {
        // Compute residual between continuous 120 Hz visual yaw/pitch and landed Cam[n].
        g_residualYaw = std::clamp(WrapPi(g_visYaw - curr.yaw), -0.10f, 0.10f);
        g_residualPitch = std::clamp(g_visPitch - curr.pitch, -0.07f, 0.07f);
      }
    }

    // Continuous dt damping & angular velocity integration in 60 Hz-tick units:
    // At 120 Hz (dt60 = 0.5), k = 1 - sqrt(1 - k60) and step = vel * 0.5,
    // matching the fixed-120Hz filter while staying rate-invariant across VRR.
    const float dt60 = std::clamp(dtSec * 60.0f, 0.05f, 2.0f);
    const float k60 = curr.modernCam ? 0.45f : 0.30f;
    const float k = 1.0f - std::pow(1.0f - k60, dt60);

    const float sx = ShapeAxisInput(subIn.rx, subIn.isMouse, false);
    const float sy = ShapeAxisInput(subIn.ry, subIn.isMouse, false);

    const bool activeTurn = (std::abs(sx) > 1e-4f || std::abs(sy) > 1e-4f ||
                             std::abs(g_velYaw) > 2e-4f || std::abs(g_velPitch) > 2e-4f);
    const bool hasResidual = (std::abs(g_residualYaw) > 1e-4f || std::abs(g_residualPitch) > 1e-4f);

    if (activeTurn) {
      float stepYaw = 0.0f;
      float stepPitch = 0.0f;
      if (subIn.isMouse) {
        // Mouse delta is a per-pass displacement (not a stick rate): apply directly
        // so total camera angle stays invariant across dt60 / VRR swings, while
        // decaying any leftover stick inertia.
        const float decay = 1.0f - k;
        g_velYaw *= decay;
        g_velPitch *= decay;
        if (std::abs(g_velYaw) < 2e-4f) g_velYaw = 0.0f;
        if (std::abs(g_velPitch) < 2e-4f) g_velPitch = 0.0f;
        stepYaw = sx * g_signedGainYaw + g_velYaw * dt60;
        stepPitch = sy * g_signedGainPitch + g_velPitch * dt60;
      } else {
        const float targetVelYaw = sx * g_signedGainYaw;
        const float targetVelPitch = sy * g_signedGainPitch;
        g_velYaw = (1.0f - k) * g_velYaw + k * targetVelYaw;
        g_velPitch = (1.0f - k) * g_velPitch + k * targetVelPitch;
        stepYaw = g_velYaw * dt60;
        stepPitch = g_velPitch * dt60;
      }

      // Gradually absorb residual directly into g_visYaw/g_visPitch over continuous dt60
      // so pass-to-pass and logic-window transitions stay monotonic with zero bounce.
      const float kRecon = 1.0f - std::pow(1.0f - 0.39f, dt60);
      const float corrYaw = g_residualYaw * kRecon;
      const float corrPitch = g_residualPitch * kRecon;
      g_residualYaw -= corrYaw;
      g_residualPitch -= corrPitch;

      g_visYaw = WrapPi(g_visYaw - corrYaw + stepYaw);
      g_visPitch = std::clamp(g_visPitch - corrPitch + stepPitch, -1.45f, 1.45f);
    } else if (hasResidual && !cut) {
      // Smoothly bleed remaining look residual into auto-follow target on stick
      // release instead of snapping abruptly back to Cam[n].
      g_velYaw = 0.0f;
      g_velPitch = 0.0f;
      const float bleed = std::pow(0.30f, dt60);
      g_residualYaw *= bleed;
      g_residualPitch *= bleed;
      if (std::abs(g_residualYaw) < 1e-4f) g_residualYaw = 0.0f;
      if (std::abs(g_residualPitch) < 1e-4f) g_residualPitch = 0.0f;
      const float baseYaw = LerpAngle(prev.yaw, curr.yaw, alpha);
      const float basePitch = prev.pitch + (curr.pitch - prev.pitch) * alpha;
      g_visYaw = WrapPi(baseYaw + g_residualYaw);
      g_visPitch = std::clamp(basePitch + g_residualPitch, -1.45f, 1.45f);
    } else {
      // No active user look input or residual: interpolate auto-follow motion.
      g_velYaw = 0.0f;
      g_velPitch = 0.0f;
      g_residualYaw = 0.0f;
      g_residualPitch = 0.0f;
      g_visYaw = LerpAngle(prev.yaw, curr.yaw, alpha);
      g_visPitch = prev.pitch + (curr.pitch - prev.pitch) * alpha;
      if (cut || alpha >= 0.999f) {
        // On Sub-Frame 1 (or a camera cut) with no user turn extrapolation,
        // write the authoritative Cam[n] matrices verbatim.
        WriteSubFrameToGuest(curr, {curr.eye.x, curr.eye.y, curr.eye.z},
                             {curr.at.x, curr.at.y, curr.at.z}, curr.fov, curr.view, curr.proj);
        return;
      }
    }

    const float renderYaw = g_visYaw;
    const float renderPitch = g_visPitch;
    const float roll120 = LerpAngle(prev.roll, curr.roll, alpha);

    const Vec3 atPrev{prev.at.x, prev.at.y, prev.at.z};
    const Vec3 atCurr{curr.at.x, curr.at.y, curr.at.z};
    const Vec3 eyePrev{prev.eye.x, prev.eye.y, prev.eye.z};
    const Vec3 eyeCurr{curr.eye.x, curr.eye.y, curr.eye.z};

    const Vec3 fwd120 = YawPitchToFwd(renderYaw, renderPitch);
    const float dist120 = prev.dist + (curr.dist - prev.dist) * alpha;
    const float fov120 = prev.fov + (curr.fov - prev.fov) * alpha;

    // Only pivot around Eye for true 1st-person aim (very small eye-to-at distance);
    // 3rd-person combat always orbits around the interpolated look-at target.
    const bool firstPersonPivot = (curr.dist < 0.6f);

    Vec3 eye120{}, at120{};
    if (firstPersonPivot) {
      eye120 = Lerp(eyePrev, eyeCurr, alpha);
      at120 = eye120 + fwd120 * dist120;
    } else {
      at120 = Lerp(atPrev, atCurr, alpha);
      eye120 = at120 - fwd120 * dist120;
    }

    Mat4 view120 = BuildLookAtNG2(eye120, at120, roll120);
    Mat4 proj120 = curr.proj;
    if (std::abs(fov120 - curr.fov) > 1e-4f && fov120 > 0.05f && curr.fov > 0.05f) {
      const float ratio = std::tan(curr.fov * 0.5f) / std::tan(fov120 * 0.5f);
      proj120.m[0][0] *= ratio;
      proj120.m[1][1] *= ratio;
    }

    WriteSubFrameToGuest(curr, eye120, at120, fov120, view120, proj120);
    return;
  }

  // Action camera / Cutscene / Locked camera mode: reset follow velocity and
  // interpolate Cam[n-1] -> Cam[n] cleanly unless a shot cut occurred.
  g_velYaw = 0.0f;
  g_velPitch = 0.0f;
  g_residualYaw = 0.0f;
  g_residualPitch = 0.0f;
  g_visYaw = curr.yaw;
  g_visPitch = curr.pitch;
  g_visInit = true;
  g_lastHandledLogicId = curr.logicFrameId;

  if (cut || alpha >= 0.999f) {
    // Final sub-frame (alpha=1.0) or camera cut: ensure Cam[n] exact matrices are loaded.
    WriteSubFrameToGuest(curr, {curr.eye.x, curr.eye.y, curr.eye.z},
                         {curr.at.x, curr.at.y, curr.at.z}, curr.fov, curr.view, curr.proj);
    return;
  }

  const Vec3 eye120 = Lerp({prev.eye.x, prev.eye.y, prev.eye.z},
                           {curr.eye.x, curr.eye.y, curr.eye.z}, alpha);
  const Vec3 at120 = Lerp({prev.at.x, prev.at.y, prev.at.z},
                          {curr.at.x, curr.at.y, curr.at.z}, alpha);
  const float fov120 = prev.fov + (curr.fov - prev.fov) * alpha;

  // Quaternion slerp handles camera Roll & steep vertical overhead angles cleanly.
  const Quat qPrev = ViewToCamQuat(prev.view);
  const Quat qCurr = ViewToCamQuat(curr.view);
  const Quat q120 = SlerpQuat(qPrev, qCurr, alpha);
  const Mat4 view120 = BuildViewFromCamQuatAndEye(q120, eye120);

  Mat4 proj120 = curr.proj;
  if (std::abs(fov120 - curr.fov) > 1e-4f && fov120 > 0.05f && curr.fov > 0.05f) {
    const float ratio = std::tan(curr.fov * 0.5f) / std::tan(fov120 * 0.5f);
    proj120.m[0][0] *= ratio;
    proj120.m[1][1] *= ratio;
  }

  WriteSubFrameToGuest(curr, eye120, at120, fov120, view120, proj120);
}

void RestoreLogicCamera() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_injectedActive || !g_curr.valid) {
    g_injectedActive = false;
    return;
  }
  const CameraSnapshot& c = g_curr;
  const uint32_t mgr = c.mgr;
  if (mgr) {
    const uint32_t slotBase = mgr + kMgrOffRenderBase + g_injectedRenderSlot * kMgrOffRenderStride;
    StoreMat4(slotBase + 0, c.view);
    StoreMat4(slotBase + 64, c.proj);
    StoreVec4(slotBase + 128, c.eye);
    StoreVec4(slotBase + 144, c.at);
    StoreVec4(slotBase + 160, c.aux);
    StoreMat4(slotBase + 176, c.auxProj0);
    StoreMat4(slotBase + 240, c.auxProj1);
    StoreFloat(slotBase + 304, c.fov);
  }
  g_injectedActive = false;
}

bool ReprojectScreenPoint(float x60, float y60, float* outX120, float* outY120) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_injectedActive || !g_curr.valid) {
    return false;
  }
  const float p00 = g_curr.proj.m[0][0];
  const float p11 = g_curr.proj.m[1][1];
  if (std::abs(p00) < 1e-4f || std::abs(p11) < 1e-4f) {
    return false;
  }

  // Reconstruct world-space ray from Cam[n] 1280x720 virtual screen coordinates.
  // NG2 view space looks along +Z (Proj.m[2][3] = +1.0f).
  const float ndcX = (x60 / 640.0f) - 1.0f;
  const float ndcY = 1.0f - (y60 / 360.0f);
  const float vx = ndcX / p00;
  const float vy = ndcY / p11;
  const float vz = 1.0f;

  const Mat4& V = g_curr.view;
  const Vec3 dirWorld{
      vx * V.m[0][0] + vy * V.m[0][1] + vz * V.m[0][2],
      vx * V.m[1][0] + vy * V.m[1][1] + vz * V.m[1][2],
      vx * V.m[2][0] + vy * V.m[2][1] + vz * V.m[2][2],
  };

  constexpr float kAimRefDist = 50.0f;
  const Vec3 pWorld{
      g_curr.eye.x + dirWorld.x * kAimRefDist,
      g_curr.eye.y + dirWorld.y * kAimRefDist,
      g_curr.eye.z + dirWorld.z * kAimRefDist,
  };

  const Vec4 clip = ProjectWorldPoint(g_subFrameViewProj, pWorld);
  if (clip.w <= 1e-4f) {
    return false;
  }
  const float nx = clip.x / clip.w;
  const float ny = clip.y / clip.w;
  if (outX120) *outX120 = (nx + 1.0f) * 640.0f;
  if (outY120) *outY120 = (1.0f - ny) * 360.0f;
  return true;
}

}  // namespace ng2::high_fps::camera
