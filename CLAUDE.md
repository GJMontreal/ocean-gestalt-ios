# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This is a clean-slate Swift/Metal iOS port of **ocean-gestalt** — a Gerstner-wave ocean simulation originally written in C++/OpenGL. The reference codebase is:

- **`~/ocean-gestalt`** — original C++/OpenGL desktop implementation. The authoritative source for algorithms, shader math, audio design, and rendering passes.

Key reference files in the C++ source:
- `src/core/OceanGestalt.cpp` — render loop and scene orchestration
- `src/core/Application.cpp` — app lifecycle
- `src/core/ReflectionPass.cpp` / `src/core/ShadowPass.cpp` / `src/core/FoamRenderPass.cpp` — render passes as discrete objects
- `src/core/Skybox.cpp`, `src/core/Prop.cpp`, `src/core/GltfModel.cpp` — scene element types
- `src/core/Drawable.cpp` / `src/core/Moveable.cpp` — rendering and physics base types
- `src/core/Mesh.cpp` / `src/core/Ocean.cpp` — ocean mesh and surface
- `src/core/GerstnerWave.cpp` — CPU Gerstner wave evaluation
- `src/core/SurfAudio.cpp` — procedural surf audio
- `src/core/Light.cpp`, `src/core/Configuration.cpp`, `src/core/Serialization.cpp` — lighting, JSON config, serialization
- `src/api/UniformState.cpp` / `src/api/UniformAnimator.cpp` — parameter state and animation
- `data/shader/` — native GLSL shaders (reference for Metal translation; ignore `data/shader/webgl/`)

## Build

The project uses **XcodeGen** to generate `OceanGestalt.xcodeproj` from `project.yml`:

```sh
xcodegen generate   # run from repo root whenever project.yml changes
```

Build and run in Xcode. The simulator supports Metal on Apple Silicon Macs; a physical device is preferred for GPU performance testing.

The `GestureCamera` Swift package is a local dependency at `~/bash-design/gesture_camera`. It ships three products — `GestureCamera` (core), `GestureCameraMetalKit`, and `GestureCameraSceneKit` — use `GestureCameraMetalKit` for this project.

## Architecture

### Scene description

The scene is driven by `data/config/scene.json`, mirroring the reference project. It describes:
- `camera` — initial position, yaw, pitch, zoom
- `light` — world-space position
- `mesh` — ocean grid size and subdivision count
- `models[]` — scene objects with file, name, position, rotation, scale, shader, floating/physics properties
- `reflection` / `shadow` — offscreen render target sizes

Nothing about the scene should be hardcoded in Swift. Adding a new floating object means editing `scene.json`, not source code.

### Modularity

Follow the C++ project's approach: each major concern is its own type. The render pipeline setup is inherently different in Metal (pipeline state objects are compiled upfront), but the draw-time logic and scene organisation should still be decomposed:

- **`Moveable`** — position, floating, tethered flags, and movement direction. Physics/position state only; no rendering knowledge.
- **`Drawable` protocol** — encodes its own draw calls into an `MTLRenderCommandEncoder`. Owns its mesh buffers and holds a reference to its pipeline state.
- **`Prop`** — composes a `Moveable` and a `Drawable`. Runs per-frame wave displacement, surface normal alignment, and angular dynamics (spin torque, angular damping, torsional stiffness) before handing off to the drawable. Mirrors `src/core/Prop.cpp`.
- **Render passes** (`ReflectionPass`, etc.) — own their textures and descriptors; expose an `encode(into:)` method.
- **Scene** — loads `scene.json`, owns the element and pass lists, and sequences rendering each frame.
- **`UniformState`** / **`UniformAnimator`** — parameter state and animated transitions, separate from rendering.
- **`SurfAudio`** — audio, entirely separate from rendering.

Avoid letting any single file become a catch-all. If a type is doing more than one thing (managing geometry, driving uniforms, *and* encoding draw calls), split it.

### Uniform structs

C structs in `ShaderTypes.h` are shared between Swift and Metal via the bridging header. `simd_float3` fields must sit at 16-byte-aligned offsets — add explicit padding fields when needed, and verify alignment whenever structs are modified.

### Wave evaluation (Gerstner)

Gerstner waves are evaluated on both CPU (Swift) and GPU (Metal) and **must stay in sync**. The CPU path drives floating object positioning and audio steepness. Always use pure displacement (no base position added) when computing gradients or steepness — adding the base XZ position contaminates the gradient with a constant term.

### Reflection pass

View matrix for the reflection pass is `viewMatrix * reflectY` where `reflectY` scales column 1 Y by -1 — **not** a second `lookAt` from a reflected eye position.

### Camera

`GestureCameraController` (from `GestureCameraMetalKit` at `~/bash-design/gesture_camera`) handles pan/pinch/gyro. Camera floating is applied by offsetting the eye position by the vertical wave displacement at the camera's XZ — the look direction is unchanged.

### Audio

Procedural stereo surf via `AVAudioEngine`: bandpass-filtered white noise driven by wave steepness sampled at two points flanking the camera (L/R, ~5 units apart).

## Key Conventions

- `project.yml` is the source of truth for project settings; never edit `.xcodeproj` directly
- `data/config/scene.json` is the source of truth for the scene; never hardcode scene content in Swift
- Commit after every meaningful code change; no need to push
- Confirm approach before editing code
- Wireframe grid sits just below the surface (model matrix shifted -0.2 Y) so wave crests naturally occlude it
