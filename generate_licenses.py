import json
import os
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class License:
    name: str
    version: Optional[str]
    license: Optional[str]
    text: str


def generate_licenses() -> List[License]:
    licenses: List[License] = []

    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/Hiroshiba/voicevox/main/LGPL_LICENSE"
    ) as res:
        licenses.append(
            License(
                name="voicevox",
                version="0.7.5-modified-by-shirowanisan",
                license="GNU LESSER GENERAL PUBLIC LICENSE Version 3",
                text=res.read().decode(),
            )
        )
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/Hiroshiba/voicevox_engine/master/LGPL_LICENSE"
    ) as res:
        licenses.append(
            License(
                name="voicevox_engine",
                version="0.7.5-modified-by-shirowanisan",
                license="GNU LESSER GENERAL PUBLIC LICENSE Version 3",
                text=res.read().decode(),
            )
        )
    # openjtalk
    # https://sourceforge.net/projects/open-jtalk/files/Open%20JTalk/open_jtalk-1.11/
    licenses.append(
        License(
            name="Open JTalk",
            version="1.11",
            license="Modified BSD license",
            text=Path("docs/licenses/open_jtalk/COPYING").read_text(),
        )
    )
    licenses.append(
        License(
            name="MeCab",
            version=None,
            license="Modified BSD license",
            text=Path("docs/licenses/open_jtalk/mecab/COPYING").read_text(),
        )
    )
    licenses.append(
        License(
            name="NAIST Japanese Dictionary",
            version=None,
            license="Modified BSD license",
            text=Path("docs/licenses//open_jtalk/mecab-naist-jdic/COPYING").read_text(),
        )
    )
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/r9y9/pyopenjtalk/master/pyopenjtalk/htsvoice/LICENSE_mei_normal.htsvoice"  # noqa: B950
    ) as res:
        licenses.append(
            License(
                name='HTS Voice "Mei"',
                version=None,
                license="Creative Commons Attribution 3.0 license",
                text=res.read().decode(),
            )
        )

    # world
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/mmorise/World/master/LICENSE.txt"
    ) as res:
        licenses.append(
            License(
                name="world",
                version=None,
                license="Modified BSD license",
                text=res.read().decode(),
            )
        )

    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/kxxoling/PTable/master/COPYING"
    ) as res:
        licenses.append(
            License(
                name="PTable",
                version="1.9.0",
                license="BSD License (BSD (3 clause))",
                text=res.read().decode(),
            )
        )

    # Python
    python_version = "3.8.10"
    with urllib.request.urlopen(
        f"https://raw.githubusercontent.com/python/cpython/v{python_version}/LICENSE"
    ) as res:
        licenses.append(
            License(
                name="Python",
                version=python_version,
                license="Python Software Foundation License",
                text=res.read().decode(),
            )
        )

    # pip
    licenses_json = json.loads(
        subprocess.run(
            "pip-licenses "
            "--from=mixed "
            "--format=json "
            "--with-urls "
            "--with-license-file "
            "--no-license-path ",
            shell=True,
            capture_output=True,
            check=True,
            env=os.environ,
        ).stdout.decode()
    )
    for license_json in licenses_json:
        license = License(
            name=license_json["Name"],
            version=license_json["Version"],
            license=license_json["License"],
            text=license_json["LicenseText"],
        )
        # FIXME: assert license type
        if license.text == "UNKNOWN":
            if license.name.lower() == "core" and license.version == "0.0.0":
                continue
            elif license.name.lower() == "nuitka":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/Nuitka/Nuitka/develop/LICENSE.txt"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "pyopenjtalk":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/r9y9/pyopenjtalk/master/LICENSE.md"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "python-multipart":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/andrew-d/python-multipart/master/LICENSE.txt"  # noqa: B950
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "romkan":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/soimort/python-romkan/master/LICENSE"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "distlib":
                with urllib.request.urlopen(
                    "https://bitbucket.org/pypa/distlib/raw/7d93712134b28401407da27382f2b6236c87623a/LICENSE.txt"  # noqa: B950
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "espnet":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/espnet/espnet/master/LICENSE"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "parallel-wavegan":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/kan-bayashi/ParallelWaveGAN/master/LICENSE"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "sentencepiece":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/google/sentencepiece/master/LICENSE"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "editdistance":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/roy-ht/editdistance/master/LICENSE"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "espnet-tts-frontend":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/espnet/espnet_tts_frontend/master/tacotron_cleaner/LICENSE"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "kaldiio":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/nttcslab-sp/kaldiio/v2.17.2/LICENSE"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "matplotlib":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/matplotlib/matplotlib/v3.1.0/LICENSE/LICENSE"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "protobuf":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/protocolbuffers/protobuf/v3.19.1/LICENSE"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "tensorboard-data-server":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/tensorflow/tensorboard/data-server-v0.6.1/LICENSE"
                ) as res:
                    license.text = res.read().decode()
            elif license.name.lower() == "torch-complex":
                with open('docs/licenses/torch_complex/for-torch-complex-license.txt') as f:
                    license.text = '\n'.join(f.readlines())
            else:
                # ライセンスがpypiに無い
                raise Exception(f"No License info provided for {license.name}")
        licenses.append(license)

    # OpenBLAS
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/xianyi/OpenBLAS/develop/LICENSE"
    ) as res:
        licenses.append(
            License(
                name="OpenBLAS",
                version=None,
                license="BSD 3-clause license",
                text=res.read().decode(),
            )
        )

    # libsndfile-binaries
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/bastibe/libsndfile-binaries/84cb164928f17c7ca0c1e5c40342c20ce2b90e8c/COPYING"  # noqa: B950
    ) as res:
        licenses.append(
            License(
                name="libsndfile-binaries",
                version="1.0.28",
                license="LGPL-2.1 license",
                text=res.read().decode(),
            )
        )

    # libogg
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/xiph/ogg/v1.3.2/COPYING"
    ) as res:
        licenses.append(
            License(
                name="libogg",
                version="1.3.2",
                license="BSD 3-clause license",
                text=res.read().decode(),
            )
        )

    # libvorbis
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/xiph/vorbis/v1.3.5/COPYING"
    ) as res:
        licenses.append(
            License(
                name="libvorbis",
                version="1.3.5",
                license="BSD 3-clause license",
                text=res.read().decode(),
            )
        )

    # libflac
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/xiph/flac/1.3.2/COPYING.Xiph"
    ) as res:
        licenses.append(
            License(
                name="FLAC",
                version="1.3.2",
                license="Xiph.org's BSD-like license",
                text=res.read().decode(),
            )
        )

    return licenses


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str)
    args = parser.parse_args()

    output_path = args.output_path

    licenses = generate_licenses()

    # dump
    out = Path(output_path).open("w") if output_path else sys.stdout
    json.dump(
        [asdict(license) for license in licenses],
        out,
    )
