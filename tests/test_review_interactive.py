import csv
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import Mock

import pytest

from smart_helmet.dataset import review_interactive as ui
from smart_helmet.dataset.cleaning_review import read_csv, validate_decisions
from smart_helmet.dataset.cleaning_visuals import render_html


def make_review(directory):
    queue = []
    for rid, priority in [('P0_b','P0'),('P0_a','P0'),('P0_c','P0'),('P1_x','P1'),('P2_x','P2'),('P3_x','P3')]:
        queue.append(dict(review_id=rid,priority=priority,issue_type='CROSS_SPLIT_NEAR_DUPLICATE',
                          visualization_path=f'review_images/{rid}.png'))
    decisions = [dict(review_id=r['review_id'],decision='PENDING',notes='existing note, giữ nguyên') for r in queue]
    for name,rows in [('review_queue.csv',queue),('review_decisions.csv',decisions)]:
        with (directory/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return queue, decisions


@pytest.mark.parametrize('choice,expected', [
    ('1','KEEP_BOTH'),('2','KEEP_A_EXCLUDE_B'),('3','KEEP_B_EXCLUDE_A'),('4','REVIEW_MORE'),
    (' s ','SKIP'),('q','QUIT'),('bad',None),('',None),('5',None),
])
def test_choice_mapping(choice,expected):
    assert ui.map_choice(choice)==expected


def test_atomic_write_order_notes_backup_and_other_priorities(tmp_path,monkeypatch):
    queue,original=make_review(tmp_path)
    path=tmp_path/'review_decisions.csv';before=path.read_bytes()
    session=ui.DecisionSession(tmp_path)
    replace=ui.os.replace;calls=[]

    def checked_replace(source,destination):
        assert Path(source).parent==tmp_path
        assert Path(destination)==path
        assert validate_decisions(queue,read_csv(source))['valid']
        assert path.read_bytes()==before
        calls.append((source,destination))
        replace(source,destination)

    monkeypatch.setattr(ui.os,'replace',checked_replace)
    session.save('P0_b','KEEP_BOTH','')
    assert len(calls)==1
    rows=read_csv(path)
    assert [r['review_id'] for r in rows]==[r['review_id'] for r in original]
    assert rows[0]['notes']==original[0]['notes']
    assert rows[1:]==original[1:]
    backup=list((tmp_path/'backups').glob('review_decisions_*.csv'))
    assert len(backup)==1 and backup[0].read_bytes()==before
    monkeypatch.setattr(ui.os,'replace',replace)
    session.save('P0_a','REVIEW_MORE','new note, line 1\nline 2')
    assert len(list((tmp_path/'backups').glob('*.csv')))==1
    assert read_csv(path)[1]['notes']=='new note, line 1\nline 2'
    assert read_csv(path)[3:]==original[3:]
    with pytest.raises(ValueError):session.save('P1_x','KEEP')


def test_invalid_temporary_csv_never_overwrites(tmp_path,monkeypatch):
    make_review(tmp_path);session=ui.DecisionSession(tmp_path)
    original=session.path.read_bytes()
    def bad_writer(path,rows,fields):
        Path(path).write_text('review_id,decision,notes\nP0_b,TYPO,\n')
    monkeypatch.setattr(ui,'write_csv',bad_writer)
    with pytest.raises(ValueError):session.save('P0_b','KEEP_BOTH')
    assert session.path.read_bytes()==original
    assert not (tmp_path/'backups').exists()
    assert not list(tmp_path.glob('.review_decisions_*.tmp'))


def test_replace_failure_preserves_original(tmp_path,monkeypatch):
    make_review(tmp_path);session=ui.DecisionSession(tmp_path);original=session.path.read_bytes()
    def fail_replace(*args):raise OSError('simulated file lock')
    monkeypatch.setattr(ui.os,'replace',fail_replace)
    with pytest.raises(OSError):session.save('P0_b','KEEP_BOTH')
    assert session.path.read_bytes()==original
    assert session.backup.read_bytes()==original
    assert not list(tmp_path.glob('.review_decisions_*.tmp'))


def test_external_edit_rejected(tmp_path):
    make_review(tmp_path);session=ui.DecisionSession(tmp_path)
    session.path.write_bytes(session.path.read_bytes()+b'\n')
    edited=session.path.read_bytes()
    with pytest.raises(ValueError,match='External edit'):session.save('P0_b','KEEP_BOTH')
    assert session.path.read_bytes()==edited


def test_skip_quit_invalid_input_no_write(tmp_path):
    make_review(tmp_path);path=tmp_path/'review_decisions.csv';before=path.read_bytes()
    inputs=iter(['typo','s','q']);log=[];opener=Mock()
    ui.review_cases(tmp_path,input_fn=lambda prompt:next(inputs),output=log.append,image_opener=opener)
    assert any('Invalid choice' in s for s in log)
    assert path.read_bytes()==before
    assert not (tmp_path/'backups').exists()
    opener.assert_not_called()


def test_resume_pending_and_image_mock(tmp_path):
    queue,_=make_review(tmp_path);log=[]
    inputs=iter(['1','','q'])
    ui.review_cases(tmp_path,input_fn=lambda prompt:next(inputs),output=log.append)
    assert read_csv(tmp_path/'review_decisions.csv')[0]['decision']=='KEEP_BOTH'
    opener=Mock();log=[];inputs=iter(['4','','q'])
    ui.review_cases(tmp_path,pending_only=True,open_image=True,input_fn=lambda prompt:next(inputs),output=log.append,image_opener=opener)
    assert not any('review_id: P0_b' in s for s in log)
    assert any('P0 REVIEW 1 / 2' in s for s in log)
    opener.assert_any_call((tmp_path/'review_images/P0_a.png').resolve())
    rows=read_csv(tmp_path/'review_decisions.csv')
    assert rows[0]['decision']=='KEEP_BOTH' and rows[1]['decision']=='REVIEW_MORE'
    result=validate_decisions(queue,rows)
    assert result['priorities']['P0']==dict(total=3,completed=1,pending=1,review_more=1)
    assert not result['ready_to_apply_cleaning']
    assert len(list((tmp_path/'backups').glob('*.csv')))==2


def test_eof_during_notes_does_not_save(tmp_path):
    make_review(tmp_path);before=(tmp_path/'review_decisions.csv').read_bytes();count=0
    def interrupted(prompt):
        nonlocal count
        count+=1
        if count==1:return '1'
        raise EOFError
    ui.review_cases(tmp_path,input_fn=interrupted,output=lambda s:None)
    assert (tmp_path/'review_decisions.csv').read_bytes()==before


def test_open_report_existing_and_missing(tmp_path,monkeypatch):
    browser=Mock(return_value=True);monkeypatch.setattr(ui.webbrowser,'open',browser);log=[]
    assert not ui.open_report(tmp_path,log.append)
    assert 'prepare_cleaning_review.py' in log[0]
    browser.assert_not_called()
    report=tmp_path/'review_report/index.html';report.parent.mkdir();report.write_text('<html></html>')
    assert ui.open_report(tmp_path,log.append)
    browser.assert_called_once_with(report.resolve().as_uri(),new=2)


def test_os_image_opener_mock(tmp_path,monkeypatch):
    image=tmp_path/'sample.png';image.touch()
    opener=Mock()
    if ui.os.name=='nt':
        monkeypatch.setattr(ui.os,'startfile',opener)
        ui.open_visualization(image)
        opener.assert_called_once_with(str(image.resolve()))
    else:
        monkeypatch.setattr(ui.webbrowser,'open',opener)
        ui.open_visualization(image)
        opener.assert_called_once_with(image.resolve().as_uri())
    with pytest.raises(FileNotFoundError):ui.open_visualization(tmp_path/'missing.png')


def test_html_readable_and_balanced():
    row=dict(review_id='P0_a',priority='P0',members=['train/<image>.jpg'],suggested_action='KEEP_BOTH',
             reason='A & B',evidence='test',visualization_path='review_images/P0_a.png')
    html=render_html([row],{})
    assert html.startswith('<!doctype html>\n')
    assert '\n  <head>\n' in html and '\n        <article ' in html
    assert len(html.splitlines())>50
    assert '&lt;image&gt;' in html and 'A &amp; B' in html
    class Balanced(HTMLParser):
        def __init__(self):super().__init__();self.stack=[];self.images=[]
        def handle_starttag(self,tag,attrs):
            if tag not in {'meta','img','br'}:self.stack.append(tag)
            if tag=='img':self.images.append(dict(attrs)['src'])
        def handle_endtag(self,tag):assert self.stack.pop()==tag
    parser=Balanced();parser.feed(html)
    assert not parser.stack
    assert parser.images==['../review_images/P0_a.png']
