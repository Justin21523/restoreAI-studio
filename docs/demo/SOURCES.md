# Demo source assets

The archive portrait and vintage camera masters were generated specifically for
RestorAI Studio with the built-in OpenAI image generation tool. They depict no
real person, product, or trademark. The original generated files are retained in
`docs/demo/sources/`.

## Archive portrait master

- Use: CodeFormer face restoration plus Real-ESRGAN 2×.
- Prompt intent: a realistic mid-century Taiwanese archive portrait with one
  unobstructed adult face, natural skin texture, period clothing, soft window
  light, no text, logos, borders, damage, or watermark.

## Product detail master

- Use: Real-ESRGAN 4×.
- Prompt intent: a realistic unbranded vintage mechanical camera on dark walnut,
  rich metal/leather/wood micro-texture, three-quarter close-up, no readable
  labels, logos, hands, or watermark.

## City motion

The short city scene is generated deterministically by
`scripts/build_demo_scenarios.py`. It contains a synthetic skyline, moving car,
rain, road motion, and a generated audio tone. No external media is used.

The same script creates degraded inputs and real GPU outputs. It records timings,
model checksums, dimensions, FPS, and audio preservation in the scenario manifest.
