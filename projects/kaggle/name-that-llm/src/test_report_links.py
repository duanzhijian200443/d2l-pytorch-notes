from pathlib import Path
import unittest

from report_links import rebase_local_links


class ReportLinksTests(unittest.TestCase):
    def test_moved_reports_keep_cross_links_and_rebase_evidence(self):
        root = Path(__file__).resolve().parents[1]
        archive = root / 'archive/research-notes'
        moved = {root/'one.md': archive/'one.md', root/'two.md': archive/'two.md'}
        original = '[peer](two.md) [data](reviews/result.json#details) [home](README.md)'
        actual = rebase_local_links(original, root, archive, moved)
        self.assertEqual(actual, '[peer](two.md) [data](../../reviews/result.json#details) [home](../../README.md)')
        reverse = {v: k for k, v in moved.items()}
        self.assertEqual(rebase_local_links(actual, archive, root, reverse), original)

    def test_urls_fragments_and_markdown_text_are_unchanged(self):
        text = '[web](https://example.com/page?q=1#x) [self](#section) [mail](mailto:test@example.com)\r\n正文保持。'
        self.assertEqual(rebase_local_links(text, Path.cwd(), Path.cwd()/'archive'), text)


if __name__ == '__main__':
    unittest.main()
