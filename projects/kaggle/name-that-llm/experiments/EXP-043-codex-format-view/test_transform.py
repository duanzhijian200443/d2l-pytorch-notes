import unittest
from transform import transform

class TransformTests(unittest.TestCase):
    def test_cross_line_bold_keeps_body(self):
        r=transform('Before **THE LIGHTHOUSE\nNorth Head** after.')
        self.assertEqual(r['text'],'Before THE LIGHTHOUSE North Head after.')
        self.assertEqual(r['bold_pairs'],1)
    def test_plain_line_breaks_and_existing_spaces(self):
        self.assertEqual(transform('one\r\n\r\ntwo  \n three')['text'],'one two    three')
    def test_inline_code_and_single_emphasis_preserved(self):
        r=transform('Use `x ** 2` and *gentle* **words**.')
        self.assertEqual(r['text'],'Use `x ** 2` and *gentle* words.')
        self.assertFalse(r['ambiguous_double_star_runs'])
    def test_code_inside_bold(self):
        self.assertEqual(transform('**Use `x` here**')['text'],'Use `x` here')
    def test_literal_dunder_preserved(self):
        self.assertEqual(transform('__init__ and __bold__')['text'],'__init__ and __bold__')
    def test_unpaired_and_triple_are_flagged(self):
        for value in ('2 ** 3','**unfinished','***italic bold***',r'\**literal\**'):
            r=transform(value);self.assertEqual(r['text'],value);self.assertTrue(r['ambiguous_double_star_runs'])
    def test_one_letter_and_punctuation(self):
        self.assertEqual(transform('(**A**), **B**!')['text'],'(A), B!')
    def test_headings_lists_preserved(self):
        self.assertEqual(transform('# Heading\n- **Item**\n2. Next')['text'],'# Heading - Item 2. Next')
    def test_idempotent(self):
        raw='Email:\n\nHello **Sam**,\nUse *care* and `x**2`.'
        result=transform(raw);again=transform(result['text'])
        self.assertEqual(again['text'],result['text']);self.assertFalse(again['edits'])

if __name__=='__main__':unittest.main()
