#!/usr/bin/env python3
"""GhostReel OpenTimelineIO sidecar CLI (plan §4a, D14)."""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

try:
    import opentimelineio as otio
except ImportError as e:
    print(json.dumps({"error": f"OpenTimelineIO not installed: {e}"}))
    sys.exit(1)


def convert_cmd(args):
    try:
        timeline = otio.adapters.read_from_file(args.input)
        otio.adapters.write_to_file(timeline, args.output, adapter_name=args.adapter)
    except Exception as e:
        sys.stderr.write(f"Conversion error: {e}\n")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


def validate_cmd(args):
    try:
        timeline = otio.adapters.read_from_file(args.input)
        clips = len(list(timeline.find_clips()))
        dur = timeline.duration()
        duration_s = dur.to_seconds()
        rate = float(dur.rate)

        tracks = []
        for track in timeline.tracks:
            kind_str = (
                "Video"
                if track.kind == otio.schema.TrackKind.Video
                else "Audio"
                if track.kind == otio.schema.TrackKind.Audio
                else str(track.kind)
            )
            tracks.append(
                {
                    "name": track.name,
                    "kind": kind_str,
                    "items": len(track),
                    "clips": len(list(track.find_clips())),
                    "duration_s": track.duration().to_seconds(),
                }
            )

        missing_media = []
        for clip in timeline.find_clips():
            ref = clip.media_reference
            if ref and hasattr(ref, "target_url") and ref.target_url:
                url = ref.target_url
                if url.startswith("file://"):
                    parsed = urllib.parse.urlparse(url)
                    # url2pathname already percent-decodes; don't unquote twice.
                    path = urllib.request.url2pathname(parsed.path)
                    # url2pathname on Windows converts /C:/foo to C:\foo
                    if path and not os.path.exists(path) and path not in missing_media:
                        missing_media.append(path)

        res = {
            "clips": clips,
            "duration_s": duration_s,
            "rate": rate,
            "tracks": tracks,
            "missing_media": missing_media,
        }
        print(json.dumps(res))
    except Exception as e:
        sys.stderr.write(f"Validation error: {e}\n")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GhostReel OTIO sidecar")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_p = subparsers.add_parser("convert", help="Convert between timeline formats")
    convert_p.add_argument("input", help="Input timeline file")
    convert_p.add_argument("output", help="Output timeline file")
    convert_p.add_argument(
        "--adapter",
        default="fcp_xml",
        choices=["fcp_xml", "otio_json"],
        help="Adapter to write (default: fcp_xml)",
    )
    convert_p.set_defaults(func=convert_cmd)

    validate_p = subparsers.add_parser("validate", help="Validate and summarize a timeline file")
    validate_p.add_argument("input", help="Timeline file to validate")
    validate_p.set_defaults(func=validate_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
