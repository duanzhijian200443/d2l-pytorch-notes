"""Narrow, loss-audited format view for frozen Codex passages.

Not a general Markdown renderer. Remove ordinary paired **strong** delimiters
outside inline code, preserve body characters, and replace CR/LF runs by spaces.
Ambiguous unprotected double-star runs are reported for rejection, not guessed.
"""
import re

STRONG=re.compile(r'(?<![\w\\*])\*\*(?!\*)(?P<body>[^\s*](?:[^*]*?[^\s*])?)\*\*(?![\w*])')

def code_spans(text):
    spans=[];at=0
    while at<len(text):
        m=re.search(r'`+',text[at:])
        if not m:break
        start=at+m.start();end=at+m.end();delimiter=text[start:end]
        if start and text[start-1]=='\\':at=end;continue
        # A missing closer is protected conservatively through the end.
        match=re.search(r'(?<!`)'+re.escape(delimiter)+r'(?!`)',text[end:])
        stop=len(text) if match is None else end+match.end()
        spans.append((start,stop));at=stop
    return spans

def transform(text):
    if not isinstance(text,str):raise TypeError('text must be str')
    protected=code_spans(text)
    inside=lambda start,end:any(start<b and end>a for a,b in protected)
    edits=[];pairs=[]
    for m in STRONG.finditer(text):
        left=(m.start(),m.start()+2);right=(m.end()-2,m.end())
        if inside(*left) or inside(*right):continue
        pairs.append((m.start(),m.end()))
        for start,end in (left,right):
            edits.append({'start':start,'end':end,'before':text[start:end],'after':'','kind':'strong_delimiter'})
    removed={(e['start'],e['end']) for e in edits}
    ambiguous=[]
    for m in re.finditer(r'\*{2,}',text):
        if not inside(m.start(),m.end()) and (m.start(),m.end()) not in removed:
            ambiguous.append({'start':m.start(),'end':m.end(),'value':m.group()})
    # Preserve all existing spaces, tabs and non-format punctuation. Only actual
    # line separator characters are replaced; words cannot be glued together.
    for m in re.finditer(r'[\r\n]+',text):
        edits.append({'start':m.start(),'end':m.end(),'before':m.group(),'after':' ','kind':'line_break'})
    edits.sort(key=lambda e:e['start'])
    if any(a['end']>b['start'] for a,b in zip(edits,edits[1:])):raise ValueError('Overlapping edits')
    output=[];last=0
    for e in edits:
        assert text[e['start']:e['end']]==e['before']
        output.extend((text[last:e['start']],e['after']));last=e['end']
    output.append(text[last:]);derived=''.join(output)
    assert '\n' not in derived and '\r' not in derived
    # Remove only recorded delimiters from the reference, then check all body
    # characters including literal single stars, underscores and backticks.
    reference=text
    for e in reversed(edits):
        if e['kind']=='strong_delimiter':reference=reference[:e['start']]+reference[e['end']:]
    assert re.sub(r'\s','',reference)==re.sub(r'\s','',derived)
    assert re.findall(r'\w+',text)==re.findall(r'\w+',derived), 'Word boundaries changed'
    return {'text':derived,'edits':edits,'bold_pairs':len(pairs),
        'newline_runs':sum(e['kind']=='line_break' for e in edits),
        'ambiguous_double_star_runs':ambiguous,'code_spans':protected}
