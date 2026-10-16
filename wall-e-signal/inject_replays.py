#re-injects the exported traces into index.html's inline json blocks.
#the frontend reads replays from <script id="replay1/replay2/replaypool"> inside
#  python export_replay.py   # output/replay_1.json, replay_2.json
#  python export_pool.py      # output/pool.json
#  python inject_replays.py   # splices all three into index.html

import re, os, sys

HTML = 'index.html'
JOBS = [('replay1', 'output/replay_1.json'),
        ('replay2', 'output/replay_2.json'),
        ('replaypool', 'output/pool.json')]


def block(html, sid):
    return re.compile(
        r'(<script[^>]*id="' + re.escape(sid) + r'"[^>]*>)(.*?)(</script>)',
        re.DOTALL).search(html)


def main():
    if not os.path.exists(HTML):
        sys.exit(f"{HTML} not found - run this from the project root")
    html = open(HTML, encoding='utf-8').read()

    for sid, jpath in JOBS:
        if not os.path.exists(jpath):
            print(f"skip #{sid} ({jpath} not found)")
            continue
        payload = open(jpath, encoding='utf-8').read().strip()
        m = block(html, sid)
        if m:
            html = html[:m.start(2)] + payload + html[m.end(2):]
        else:
            # create the block right after replay2's closing tag
            anchor = block(html, 'replay2')
            if not anchor:
                sys.exit("can't find replay2 block to anchor the new one")
            tag = f'\n<script type="application/json" id="{sid}">{payload}</script>'
            html = html[:anchor.end(3)] + tag + html[anchor.end(3):]
        print(f"injected {jpath} -> #{sid} ({len(payload)/1024:.0f}KB)")

    open(HTML, 'w', encoding='utf-8').write(html)
    print(f"updated {HTML}")


if __name__ == '__main__':
    main()