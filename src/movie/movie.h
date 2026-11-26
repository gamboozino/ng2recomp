// ng2 - Ninja Gaiden II static recompilation
//
// Movies: the game's XMedia2 player replaced by host decoding
// (movie_hooks.cpp, host_movie.cpp).

#pragma once

#include <cstdint>
#include <filesystem>

namespace ng2::movie {

// Where "game:\" movie paths point.
void SetGameDataRoot(const std::filesystem::path& root);

// A movie is playing on the host player (one that hasn't ended).
bool AnyPlaying();

// True when a full-screen replacement movie (rendered via +80 render) is playing,
// excluding looping in-world texture billboards (e.g. aurora12.wmv).
bool AnyFullscreenPlaying();

// The physical address is one of the textures a movie's frames are written
// to (its Y, U or V plane): a draw of a movie the game shows itself, such as
// a scroll's (video.cpp keeps it with the rest of the 2D at other aspects).
bool IsFrameTexture(uint32_t physical);

}  // namespace ng2::movie
