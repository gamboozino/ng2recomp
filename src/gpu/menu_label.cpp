// ng2 - Ninja Gaiden II static recompilation
//
// The title menu's "Xbox LIVE Marketplace" picture becomes "RECOMP SETTINGS"
// and the unused "EXIT DEMO" one "EXIT GAME" (src/recomp_settings.cpp), made
// from the game's own letters: the other
// entries' pictures share one 8-bit atlas, one 90-pixel band per entry, in
// the same font. Each word there is split into its letters at the columns
// with the fewest bright pixels, and the new text is laid out from those
// letters with their outline and glow, so it takes the menu's palettes like
// the rest. Nothing of the game's art is stored here.

#include "menu_label.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#define XXH_INLINE_ALL
#include <xxhash.h>

namespace ng2::video {
namespace {

constexpr uint32_t kWidth = 520;
constexpr uint32_t kBand = 90;
constexpr uint64_t kAtlasHash = 0x3737e258b350fa36;   // 520x990, the entries' atlas
constexpr uint64_t kLabelHash = 0xf8fb387ec4883ffc;   // 520x90, Xbox LIVE Marketplace
// The other languages' atlases and Marketplace labels (Japanese keeps the
// English label; Portuguese and Simplified Chinese use the English ones).
constexpr uint64_t kOtherAtlases[] = {
    0xd09f6e653f0c44df,  // Japanese
    0x7ba39c742e71ab0f,  // German
    0x0bf716435014362e,  // French
    0xe34071b489a2d830,  // Spanish
    0x4d6f66ffa5fd5392,  // Italian
    0xeb91498647389ab9,  // Korean
    0x298f406c6f391369,  // Traditional Chinese
    0x24ae84c08b328168,  // Polish
    0xccf0c19dcdd7dcb0,  // Russian
};
constexpr uint64_t kOtherLabels[] = {
    0x4774ba1f7e83c512,  // German
    0x9ab60305171f1595,  // French
    0x6f37d128acb6719b,  // Spanish
    0x214c148a5445623e,  // Italian
    0xa75b722fe3c99c52,  // Korean
    0x6a908aa5f96e35ce,  // Traditional Chinese
    0x2bb41db4e7d2702a,  // Polish
    0xef9883b77fd3105f,  // Russian
};
template <size_t N>
bool OneOf(uint64_t hash, const uint64_t (&list)[N]) {
  return std::find(std::begin(list), std::end(list), hash) != std::end(list);
}
constexpr std::string_view kText = "RECOMP SETTINGS";
// The atlas's first entry, EXIT DEMO, isn't used by the retail menu: it
// becomes the exit entry's label (src/recomp_settings.cpp).
constexpr std::string_view kExitText = "EXIT GAME";
// The atlas's entries, top to bottom.
constexpr std::string_view kWords[] = {
    "EXIT DEMO", "NEW GAME", "LOAD GAME", "LEADERBOARDS", "NINJA CINEMA", "OPTIONS", "MISSIONS",
    "PATH OF THE ACOLYTE", "PATH OF THE WARRIOR", "PATH OF THE MENTOR", "PATH OF THE MASTER NINJA"};
constexpr uint8_t kClear = 255;  // outside the picture
constexpr uint8_t kPlate = 0;    // the dark plate behind the words
constexpr uint32_t kSpace = 16;  // columns between words

std::vector<uint8_t> g_atlas;  // the English atlas's pixels, once seen
bool g_atlasMissing = false;   // another language's, and the English one not found

struct Glyph {
  uint32_t band, begin, end;  // columns
};

bool IsLetter(uint8_t v) {
  return v != kClear && v != kPlate && (v <= 70 || v >= 110);
}

// The columns of each of the atlas's words, and of each letter that splits
// cleanly from its word.
std::map<std::string, Glyph, std::less<>> FindGlyphs() {
  std::map<std::string, Glyph, std::less<>> glyphs;
  // EXIT DEMO last: its letters split less cleanly than the others'.
  for (uint32_t i = 1; i <= std::size(kWords); i++) {
    uint32_t band = i % std::size(kWords);
    const uint8_t* pixels = g_atlas.data() + band * kBand * kWidth;
    // Words: runs of columns holding letter pixels.
    std::vector<std::pair<uint32_t, uint32_t>> runs;
    bool inside = false;
    for (uint32_t x = 0; x <= kWidth; x++) {
      bool ink = false;
      for (uint32_t y = 0; x < kWidth && y < kBand && !ink; y++) {
        ink = IsLetter(pixels[y * kWidth + x]);
      }
      if (ink && !inside) {
        runs.emplace_back(x, x);
      } else if (!ink && inside) {
        runs.back().second = x;
      }
      inside = ink;
    }
    // Stray columns (EXIT DEMO has one at its left edge) aren't words.
    std::erase_if(runs, [](const auto& run) { return run.second - run.first < 4; });
    std::vector<std::string_view> parts;
    for (std::string_view words = kWords[band]; !words.empty();) {
      size_t space = words.find(' ');
      parts.push_back(words.substr(0, space));
      words = space == std::string_view::npos ? std::string_view() : words.substr(space + 1);
    }
    if (runs.size() != parts.size()) {
      continue;
    }
    for (size_t w = 0; w < parts.size(); w++) {
      auto [begin, end] = runs[w];
      glyphs.try_emplace(std::string(parts[w]), Glyph{band, begin, end});
      size_t letters = parts[w].size();
      // Bright pixels per column, smoothed; the letters' joins are its
      // lowest points.
      std::vector<float> profile(end - begin);
      for (uint32_t x = begin; x < end; x++) {
        uint32_t count = 0;
        for (uint32_t y = 0; y < kBand; y++) {
          uint8_t v = pixels[y * kWidth + x];
          count += v >= 150 && v != kClear;
        }
        profile[x - begin] = float(count);
      }
      std::vector<float> smooth(profile.size());
      for (size_t i = 0; i < profile.size(); i++) {
        float left = i ? profile[i - 1] : 0.0f;
        float right = i + 1 < profile.size() ? profile[i + 1] : 0.0f;
        smooth[i] = left + 2.0f * profile[i] + right;
      }
      std::vector<uint32_t> order(smooth.size());
      for (uint32_t i = 0; i < order.size(); i++) {
        order[i] = i;
      }
      std::stable_sort(order.begin(), order.end(),
                       [&](uint32_t a, uint32_t b) { return smooth[a] < smooth[b]; });
      std::vector<uint32_t> cuts;
      for (uint32_t c : order) {
        if (cuts.size() + 1 >= letters) {
          break;
        }
        if (c < 6 || c + 6 > smooth.size()) {
          continue;
        }
        if (std::all_of(cuts.begin(), cuts.end(),
                        [&](uint32_t x) { return (c > x ? c - x : x - c) >= 10; })) {
          cuts.push_back(c);
        }
      }
      if (cuts.size() + 1 != letters) {
        continue;
      }
      std::sort(cuts.begin(), cuts.end());
      uint32_t from = begin;
      for (size_t i = 0; i < letters; i++) {
        uint32_t to = i < cuts.size() ? begin + cuts[i] : end;
        glyphs.try_emplace(std::string(1, parts[w][i]), Glyph{band, from, to});
        from = to;
      }
    }
  }
  return glyphs;
}

bool IsFace(uint8_t v) {
  return v != kClear && v >= 112;
}

// Squared distance (up to kReach, else more) from each pixel of a band's
// columns [x0, x1) to the nearest letter face in its columns [begin, end).
constexpr int kReach = 14;
std::vector<int> FaceDistance(uint32_t band, int begin, int end, int x0, int x1) {
  std::vector<int> distance(size_t(x1 - x0) * kBand, kReach * kReach + 1);
  const uint8_t* pixels = g_atlas.data() + band * kBand * kWidth;
  for (int fy = 0; fy < int(kBand); fy++) {
    for (int fx = std::max(begin, 0); fx < std::min(end, int(kWidth)); fx++) {
      if (!IsFace(pixels[fy * kWidth + fx])) {
        continue;
      }
      for (int y = std::max(0, fy - kReach); y < std::min(int(kBand), fy + kReach + 1); y++) {
        for (int x = std::max(x0, fx - kReach); x < std::min(x1, fx + kReach + 1); x++) {
          int& d = distance[size_t(y) * (x1 - x0) + (x - x0)];
          d = std::min(d, (x - fx) * (x - fx) + (y - fy) * (y - fy));
        }
      }
    }
  }
  return distance;
}

// The atlas's halo around its letters by distance from the face (its median),
// and its lowest value (below: the outline and shadow).
constexpr uint8_t kHaloBase = 0x5d;
constexpr uint8_t kHalo[] = {0x66, 0x66, 0x62, 0x60, 0x5e, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d};

// Squared distance (up to 10, else more) from each pixel of a label to its
// nearest face.
std::vector<int> FaceDistanceIn(const uint8_t* pixels) {
  std::vector<int> distance(size_t(kWidth) * kBand, 101);
  for (int fy = 0; fy < int(kBand); fy++) {
    for (int fx = 0; fx < int(kWidth); fx++) {
      if (!IsFace(pixels[fy * kWidth + fx])) {
        continue;
      }
      for (int y = std::max(0, fy - 10); y < std::min(int(kBand), fy + 11); y++) {
        for (int x = std::max(0, fx - 10); x < std::min(int(kWidth), fx + 11); x++) {
          int& d = distance[size_t(y) * kWidth + x];
          d = std::min(d, (x - fx) * (x - fx) + (y - fy) * (y - fy));
        }
      }
    }
  }
  return distance;
}

// A label into pixels (kWidth x kBand), false if a letter is missing: its
// words as the atlas has them where it does, else letter by letter.
bool Compose(std::string_view text, uint8_t* pixels) {
  auto glyphs = FindGlyphs();
  std::vector<const Glyph*> pieces;  // nullptr: a space
  for (std::string_view words = text; !words.empty();) {
    size_t space = words.find(' ');
    std::string_view word = words.substr(0, space);
    words = space == std::string_view::npos ? std::string_view() : words.substr(space + 1);
    if (!pieces.empty()) {
      pieces.push_back(nullptr);
    }
    if (auto it = glyphs.find(word); it != glyphs.end()) {
      pieces.push_back(&it->second);
      continue;
    }
    for (char c : word) {
      auto it = glyphs.find(std::string_view(&c, 1));
      if (it == glyphs.end()) {
        return false;
      }
      pieces.push_back(&it->second);
    }
  }
  uint32_t width = 0;
  for (const Glyph* piece : pieces) {
    width += piece ? piece->end - piece->begin : kSpace;
  }
  if (width > kWidth) {
    return false;
  }
  // Each letter brings its surroundings (outline, shadow, glow) up to
  // kReach from its face, but not its source word's other letters' faces,
  // nor their shadow past 5 pixels from its own face; where two placed
  // letters' surroundings meet, the nearer face's wins. Cut at fixed columns
  // instead, each letter kept its own word's shadow in a box (#66).
  std::vector<int> best(kWidth * kBand, kReach * kReach + 1);
  std::memset(pixels, kClear, kWidth * kBand);
  int x = int(kWidth - width) / 2;
  for (const Glyph* piece : pieces) {
    if (!piece) {
      x += kSpace;
      continue;
    }
    const Glyph& glyph = *piece;
    int begin = int(glyph.begin), end = int(glyph.end);
    int x0 = std::max(0, begin - kReach), x1 = std::min(int(kWidth), end + kReach);
    auto own = FaceDistance(glyph.band, begin, end, x0, x1);
    auto left = FaceDistance(glyph.band, x0 - kReach, begin, x0, x1);
    auto right = FaceDistance(glyph.band, end, x1 + kReach, x0, x1);
    const uint8_t* source = g_atlas.data() + glyph.band * kBand * kWidth;
    for (int y = 0; y < int(kBand); y++) {
      for (int sx = x0; sx < x1; sx++) {
        size_t i = size_t(y) * (x1 - x0) + (sx - x0);
        int d = own[i];
        int ox = x + sx - begin;
        uint8_t v = source[y * kWidth + sx];
        bool theirs = d > left[i] || d > right[i];
        bool shadow = v != kClear && v != kPlate && v < kHaloBase;
        if (d > kReach * kReach || (theirs && (IsFace(v) || (shadow && d > 25))) || ox < 0 ||
            ox >= int(kWidth)) {
          continue;
        }
        size_t o = size_t(y) * kWidth + ox;
        if (d < best[o]) {
          best[o] = d;
          pixels[o] = v;
        }
      }
    }
    x += end - begin;
  }
  // The halo follows the letters as the atlas's does: left empty near them,
  // the atlas's halo at that distance (its words have no plate pixels within
  // 5 of a face); past 9, the plate (a halo out to the plate's edge drew a
  // dark box around the label).
  auto faces = FaceDistanceIn(pixels);
  for (size_t i = 0; i < size_t(kWidth) * kBand; i++) {
    uint8_t& v = pixels[i];
    bool empty = v == kClear || v == kPlate;
    bool halo = v >= kHaloBase && !IsFace(v);
    if (faces[i] > 81) {
      if (halo) {
        v = kPlate;
      }
    } else if (empty) {
      int distance = int(std::lround(std::sqrt(double(faces[i]))));
      v = kHalo[std::min(distance, 9)];
    }
  }
  // One plate behind it all, as the atlas's: the letters' box and 4 to 5
  // pixels more.
  int left = kWidth, right = -1, top = kBand, bottom = -1;
  for (int y = 0; y < int(kBand); y++) {
    for (int px = 0; px < int(kWidth); px++) {
      if (IsLetter(pixels[y * kWidth + px])) {
        left = std::min(left, px), right = std::max(right, px);
        top = std::min(top, y), bottom = std::max(bottom, y);
      }
    }
  }
  left = std::max(0, left - 4), top = std::max(0, top - 4);
  right = std::min(int(kWidth) - 1, right + 5), bottom = std::min(int(kBand) - 1, bottom + 5);
  for (int y = 0; y < int(kBand); y++) {
    for (int px = 0; px < int(kWidth); px++) {
      uint8_t& v = pixels[y * kWidth + px];
      if (px < left || px > right || y < top || y > bottom) {
        v = kClear;
      } else if (v == kClear) {
        v = kPlate;
      }
    }
  }
  return true;
}

// The header picture: "RECOMP SETTINGS" from the atlas's letters, scaled to
// the header's letter size, with the plate dropped, the letter faces and
// outlines brought to the header's levels and a soft glow around them.
constexpr uint64_t kHeaderHash = 0x3e4f94c916758ea1;  // 172x58, OPTIONS
constexpr uint32_t kHeaderWidth = 172, kHeaderHeight = 58;
// The other languages' Options headers (Japanese and French keep the
// English one; Korean's isn't a picture of these).
struct OtherHeader {
  uint32_t width;
  uint64_t hash;
};
constexpr OtherHeader kOtherHeaders[] = {
    {200, 0x90036f17f3198265},  // German
    {200, 0x26dec7b46f1fc9a3},  // Spanish
    {200, 0x615b10d2d40dfb86},  // Italian
    {94, 0xb90e89d7dc897ad8},   // Traditional Chinese
    {172, 0x880058569d76308a},  // Polish
    {230, 0x6294a9b09f954a41},  // Russian
};
constexpr float kHeaderScale = 0.92f;  // header letters / atlas letters
constexpr uint32_t kHeaderTop = 14, kHeaderMargin = 12;
constexpr int kGlowRadius = 6;
// The header's palette is sorted by brightness, its alpha all over the place
// from one entry to the next (15: 237, 18: 44), so a smooth run of values came
// out speckled (#170). Ours takes only the entries the game's headers use:
// for the glow, those that make a smooth dark red ramp, by alpha (as the
// Options header's, out to about 13 pixels from the letters); for the
// outline, dark reds (kOutline).
struct GlowStep {
  uint8_t index, alpha;
};
constexpr GlowStep kGlowRamp[] = {
    {0, 0},    {1, 2},    {2, 5},    {3, 8},    {4, 11},   {5, 14},   {6, 17},   {7, 20},
    {8, 23},   {9, 26},   {10, 29},  {11, 32},  {12, 35},  {13, 38},  {14, 41},  {18, 44},
    {19, 47},  {20, 50},  {21, 53},  {24, 57},  {29, 64},  {36, 70},  {37, 76},  {40, 81},
    {50, 88},  {51, 93},  {53, 99},  {55, 104}, {64, 111},
};
constexpr float kGlowAlpha = 111.0f;  // next to the letters
constexpr float kGlowGain = 2.2f;     // (the blurred letters are about 0.5 at their edge)
// (as the Options header's next to its letters: dark reds, opaque, then half
// clear; the greyer near-opaque entries, 96 and up, came out as grey flecks)
constexpr uint8_t kOutline[] = {28, 34, 46, 60, 91, 64, 55};

uint8_t GlowIndex(float alpha) {
  const GlowStep* best = &kGlowRamp[0];
  for (const GlowStep& step : kGlowRamp) {
    if (std::fabs(float(step.alpha) - alpha) < std::fabs(float(best->alpha) - alpha)) {
      best = &step;
    }
  }
  return best->index;
}

// The in-game list's header ("IN-GAME MANUAL AND OPTIONS", 521x58), in the
// same lettering at the same size and place: ours goes in as it is, the rest
// left clear.
constexpr uint64_t kInGameHeaderHash = 0xc16cba91cd852335;
constexpr uint32_t kInGameHeaderWidth = 521;
// The other languages' in-game headers (Portuguese and Simplified Chinese use
// the English one).
constexpr OtherHeader kOtherInGameHeaders[] = {
    {600, 0x739f598f81405715},  // Japanese
    {600, 0xa88403e6cece4b2f},  // German
    {521, 0x106c505107bf91e2},  // French
    {600, 0x05e5802de7f5b991},  // Spanish
    {600, 0x200d43fdfe4ce181},  // Italian
    {600, 0x1f011959bbf09487},  // Korean
    {600, 0xa3210211f7dda0fa},  // Traditional Chinese
    {600, 0x28278a87628524b5},  // Polish
    {600, 0x477b96a506485d61},  // Russian
};

// Set by the menu, read where textures are made.
std::atomic<bool> g_headerWanted = false;
std::atomic<bool> g_inGameHeaderWanted = false;
std::atomic<uint32_t> g_headerGeneration = 0;

// A box blur, each way.
std::vector<float> Blur(const std::vector<float>& image, uint32_t width, uint32_t height) {
  std::vector<float> across(image.size()), out(image.size());
  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
      float sum = 0.0f;
      for (int d = -kGlowRadius; d <= kGlowRadius; d++) {
        int sx = int(x) + d;
        sum += sx >= 0 && sx < int(width) ? image[y * width + sx] : 0.0f;
      }
      across[y * width + x] = sum / (2 * kGlowRadius + 1);
    }
  }
  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
      float sum = 0.0f;
      for (int d = -kGlowRadius; d <= kGlowRadius; d++) {
        int sy = int(y) + d;
        sum += sy >= 0 && sy < int(height) ? across[sy * width + x] : 0.0f;
      }
      out[y * width + x] = sum / (2 * kGlowRadius + 1);
    }
  }
  return out;
}

bool MakeHeader(std::vector<uint8_t>& out, uint32_t& outWidth) {
  std::vector<uint8_t> label(kWidth * kBand);
  if (!Compose(kText, label.data())) {
    return false;
  }
  // Only the outline within 2 pixels of a face: the title's labels carry a
  // drop shadow and their source words' neighbours' edges further out (a
  // ragged strip by the P), which the Options header's letters don't have.
  {
    std::vector<uint8_t> kept = label;
    for (int y = 0; y < int(kBand); y++) {
      for (int x = 0; x < int(kWidth); x++) {
        uint8_t v = label[y * kWidth + x];
        if (!IsLetter(v) || IsFace(v)) {
          continue;
        }
        bool near = false;
        for (int dy = -2; dy <= 2 && !near; dy++) {
          for (int dx = -2; dx <= 2 && !near; dx++) {
            int nx = x + dx, ny = y + dy;
            near = dx * dx + dy * dy <= 5 && nx >= 0 && nx < int(kWidth) && ny >= 0 && ny < int(kBand) &&
                   IsFace(label[ny * kWidth + nx]);
          }
        }
        if (!near) {
          kept[y * kWidth + x] = kPlate;
        }
      }
    }
    label.swap(kept);
  }
  uint32_t x0 = kWidth, x1 = 0, y0 = kBand, y1 = 0;
  for (uint32_t y = 0; y < kBand; y++) {
    for (uint32_t x = 0; x < kWidth; x++) {
      if (IsLetter(label[y * kWidth + x])) {
        x0 = std::min(x0, x), x1 = std::max(x1, x + 1);
        y0 = std::min(y0, y), y1 = std::max(y1, y + 1);
      }
    }
  }
  if (x0 >= x1) {
    return false;
  }
  outWidth = uint32_t(std::ceil(float(x1 - x0) * kHeaderScale)) + 2 * kHeaderMargin;
  uint32_t size = outWidth * kHeaderHeight;
  std::vector<float> value(size), mask(size);
  // Bilinear over the letter pixels only.
  for (uint32_t oy = 0; oy < kHeaderHeight; oy++) {
    for (uint32_t ox = 0; ox < outWidth; ox++) {
      float sx = float(x0) + (float(ox) - kHeaderMargin + 0.5f) / kHeaderScale - 0.5f;
      float sy = float(y0) + (float(oy) - kHeaderTop + 0.5f) / kHeaderScale - 0.5f;
      int ix = int(std::floor(sx)), iy = int(std::floor(sy));
      float fx = sx - float(ix), fy = sy - float(iy);
      float sum = 0.0f, weight = 0.0f;
      for (int k = 0; k < 4; k++) {
        int x = ix + (k & 1), y = iy + (k >> 1);
        float w = ((k & 1) ? fx : 1.0f - fx) * ((k >> 1) ? fy : 1.0f - fy);
        if (x >= 0 && x < int(kWidth) && y >= 0 && y < int(kBand) &&
            IsLetter(label[y * kWidth + x])) {
          sum += float(label[y * kWidth + x]) * w;
          weight += w;
        }
      }
      if (weight > 0.5f) {
        value[oy * outWidth + ox] = sum / weight;
        mask[oy * outWidth + ox] = 1.0f;
      }
    }
  }
  // Pieces of letter under 40 pixels (a whole letter is far more): scraps of
  // a neighbour in the source lettering, as the ragged strip by the P; left
  // to the glow.
  {
    std::vector<int> piece(size, -1);
    std::vector<uint32_t> stack, members;
    for (uint32_t start = 0; start < size; start++) {
      if (mask[start] == 0.0f || piece[start] >= 0) {
        continue;
      }
      members.clear();
      stack.assign(1, start);
      piece[start] = int(start);
      while (!stack.empty()) {
        uint32_t i = stack.back();
        stack.pop_back();
        members.push_back(i);
        uint32_t x = i % outWidth, y = i / outWidth;
        const uint32_t next[4] = {x > 0 ? i - 1 : i, x + 1 < outWidth ? i + 1 : i, y > 0 ? i - outWidth : i,
                                  y + 1 < kHeaderHeight ? i + outWidth : i};
        for (uint32_t j : next) {
          if (j != i && mask[j] > 0.0f && piece[j] < 0) {
            piece[j] = int(start);
            stack.push_back(j);
          }
        }
      }
      if (members.size() < 40) {
        for (uint32_t i : members) {
          mask[i] = 0.0f;
        }
      }
    }
  }
  std::vector<float> glow = Blur(Blur(mask, outWidth, kHeaderHeight), outWidth, kHeaderHeight);
  out.resize(size);
  for (uint32_t i = 0; i < size; i++) {
    if (mask[i] > 0.0f && value[i] >= 110.0f) {  // face
      out[i] = uint8_t(std::clamp(150.0f + (value[i] - 110.0f) * (105.0f / 145.0f), 0.0f, 255.0f));
    } else if (mask[i] > 0.0f) {  // outline
      size_t k = size_t(std::clamp(value[i], 0.0f, 109.0f) / 110.0f * float(std::size(kOutline)));
      out[i] = kOutline[std::min(k, std::size(kOutline) - 1)];
    } else {
      out[i] = GlowIndex(std::min(1.0f, glow[i] * kGlowGain) * kGlowAlpha);
    }
  }
  return true;
}

}  // namespace

LabelResult RewriteMenuLabel(uint32_t width, uint32_t height, uint8_t* pixels, uint32_t* rows) {
  *rows = 0;
  if (width != kWidth || (height != kBand && height != kBand * std::size(kWords))) {
    return LabelResult::kUnchanged;
  }
  uint64_t hash = XXH3_64bits(pixels, size_t(width) * height);
  bool english = hash == kAtlasHash;
  if (english || OneOf(hash, kOtherAtlases)) {
    if (english) {
      g_atlas.assign(pixels, pixels + size_t(width) * height);
    }
    // (another language's: its own EXIT DEMO entry, in English letters)
    if (g_atlas.empty() || !Compose(kExitText, pixels)) {
      return LabelResult::kUnchanged;
    }
    *rows = kBand;
    return LabelResult::kRewritten;
  }
  if (hash != kLabelHash && !OneOf(hash, kOtherLabels)) {
    return LabelResult::kUnchanged;
  }
  if (g_atlas.empty()) {
    return g_atlasMissing ? LabelResult::kUnchanged : LabelResult::kRetry;
  }
  if (!Compose(kText, pixels)) {
    return LabelResult::kUnchanged;
  }
  *rows = kBand;
  return LabelResult::kRewritten;
}

bool NeedsEnglishAtlas(uint32_t width, uint32_t height, const uint8_t* pixels) {
  return width == kWidth && height == kBand * std::size(kWords) && g_atlas.empty() &&
         !g_atlasMissing && OneOf(XXH3_64bits(pixels, size_t(width) * height), kOtherAtlases);
}

bool SetEnglishAtlas(const uint8_t* pixels) {
  size_t size = size_t(kWidth) * kBand * std::size(kWords);
  if (XXH3_64bits(pixels, size) != kAtlasHash) {
    return false;
  }
  g_atlas.assign(pixels, pixels + size);
  return true;
}

void EnglishAtlasMissing() {
  g_atlasMissing = true;
}

void SetRecompHeader(bool on) {
  if (on != g_headerWanted) {
    g_headerWanted = on;
    g_headerGeneration++;
  }
}

void SetInGameRecompHeader(bool on) {
  if (on != g_inGameHeaderWanted) {
    g_inGameHeaderWanted = on;
    g_headerGeneration++;
  }
}

uint32_t MenuHeaderGeneration() {
  return g_headerGeneration;
}

bool IsMenuHeaderSize(uint32_t width, uint32_t height) {
  auto widthOf = [&](const OtherHeader& h) { return h.width == width; };
  return height == kHeaderHeight &&
         (width == kHeaderWidth || width == kInGameHeaderWidth ||
          std::any_of(std::begin(kOtherHeaders), std::end(kOtherHeaders), widthOf) ||
          std::any_of(std::begin(kOtherInGameHeaders), std::end(kOtherInGameHeaders), widthOf));
}

bool ReplaceMenuHeader(uint32_t width, uint32_t height, const uint8_t* pixels,
                       std::vector<uint8_t>& out, uint32_t& outWidth, MenuHeaderKind& kind) {
  kind = MenuHeaderKind::kNone;
  if (!IsMenuHeaderSize(width, height)) {
    return false;
  }
  uint64_t hash = XXH3_64bits(pixels, size_t(width) * height);
  bool other = std::any_of(std::begin(kOtherHeaders), std::end(kOtherHeaders),
                           [&](const OtherHeader& h) { return h.width == width && h.hash == hash; });
  if (hash == kHeaderHash || other) {
    kind = MenuHeaderKind::kOptions;
    return g_headerWanted && !g_atlas.empty() && MakeHeader(out, outWidth);
  }
  bool otherInGame =
      std::any_of(std::begin(kOtherInGameHeaders), std::end(kOtherInGameHeaders),
                  [&](const OtherHeader& h) { return h.width == width && h.hash == hash; });
  if (hash != kInGameHeaderHash && !otherInGame) {
    return false;
  }
  kind = MenuHeaderKind::kInGame;
  std::vector<uint8_t> header;
  uint32_t headerWidth = 0;
  if ( !g_inGameHeaderWanted || g_atlas.empty() || !MakeHeader(header, headerWidth) ||
      headerWidth > width) {
    return false;
  }
  out.assign(size_t(width) * height, 0);
  for (uint32_t y = 0; y < height; y++) {
    std::memcpy(&out[size_t(y) * width], &header[size_t(y) * headerWidth], headerWidth);
  }
  outWidth = width;
  return true;
}


}  // namespace ng2::video
