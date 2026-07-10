# Third-party notices

RestorAI Studio integrates third-party model architecture code and external weight
files. Those components are not relicensed by this project.

| Component | Use | Upstream / license |
| --- | --- | --- |
| Real-ESRGAN | Installed dependency and external x2/x4 weights | https://github.com/xinntao/Real-ESRGAN — BSD-3-Clause |
| GFPGAN | Vendored clean inference architecture and external v1.4 weight | https://github.com/TencentARC/GFPGAN — Apache-2.0 plus upstream notices |
| CodeFormer | Vendored inference architectures and external weight | https://github.com/sczhou/CodeFormer — NTU S-Lab License 1.0 |
| Practical-RIFE | Vendored v4.25 inference architecture and external weight | https://github.com/hzwer/Practical-RIFE — MIT; revision `9aff2a278b1fb5085e137b4f4b748e518bf7ab26` |
| facexlib | Installed face detection/alignment dependency and external weights | https://github.com/xinntao/facexlib — MIT |

Practical-RIFE's license text is retained at
`restorai/vendor/rife/LICENSE`. Before redistributing a build, consult each linked
upstream license and include any full license/NOTICE files its terms require.

CodeFormer is not presented as an unrestricted commercial dependency. The public
Pages demo does not run or distribute it, and operators are responsible for
confirming that their intended use complies with the NTU S-Lab License 1.0.
