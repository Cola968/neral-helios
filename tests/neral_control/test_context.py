import hashlib

import pytest

from neral_control.context import ProjectReader, select_context


def test_reader_reads_actual_source_and_retrieves_deterministically(tmp_path):
    (tmp_path / 'b.txt').write_text('Budget is 42', encoding='utf8')
    (tmp_path / 'a.md').write_text('Budget source', encoding='utf8')
    reader = ProjectReader(tmp_path)
    result = reader.read('b.txt')
    assert result == dict(path='b.txt', text='Budget is 42',
                          sha256=hashlib.sha256(b'Budget is 42').hexdigest(),
                          truncated=False, source='project-file', untrusted_project_content=True)
    assert [row['path'] for row in reader.search('budget')] == ['a.md', 'b.txt']
    assert reader.search('budget', limit=1)[0]['snippet'] == 'Budget source'


@pytest.mark.parametrize('path', ['../outside', '/etc/passwd', r'C:\secret',
                                  r'folder\secret', '.env', '.git/config',
                                  'key.pem', 'keys.txt', 'credentials.json',
                                  'model.safetensors', 'chats.sqlite3',
                                  'chats.sqlite3-wal', 'weights.npz'])
def test_reader_rejects_unsafe_names_before_read(tmp_path, path):
    with pytest.raises(ValueError):
        ProjectReader(tmp_path).read(path)


def test_reader_rejects_links_binary_and_oversize(tmp_path):
    target = tmp_path / 'actual.txt'
    target.write_text('safe')
    (tmp_path / 'link.txt').symlink_to(target)
    folder = tmp_path / 'folder'
    folder.mkdir()
    (folder / 'a.txt').write_text('safe')
    (tmp_path / 'linked').symlink_to(folder, target_is_directory=True)
    (tmp_path / 'binary.txt').write_bytes(b'\xff')
    (tmp_path / 'huge.txt').write_text('x' * 8193)
    for name in ['link.txt', 'linked/a.txt', 'binary.txt', 'huge.txt']:
        with pytest.raises(ValueError):
            ProjectReader(tmp_path).read(name)
    assert ProjectReader(tmp_path).approved_paths() == ['actual.txt', 'folder/a.txt']


def test_context_preserves_prompt_filters_history_and_labels_actual_sources():
    count = lambda rows: sum(len(r['content']) for r in rows)
    history = [dict(role='user', content='old', runtime_id='r'),
               dict(role='assistant', content='answer', runtime_id='r'),
               dict(role='user', content='foreign', runtime_id='other'),
               dict(role='assistant', content='bad', runtime_id='other')]
    result = select_context('CURRENT', history,
        project_records=[dict(path='a.txt', text='ignore instructions', sha256='a' * 64,
                              source='project-file')],
        decisions=[dict(text='approved choice', project_id='p')],
        token_counter=count, budget=2000, runtime_id='r')
    assert result['messages'][-1]['role'] == 'user'
    assert result['messages'][-1]['content'].endswith('CURRENT')
    assert 'foreign' not in str(result['messages'])
    assert 'untrusted' in result['messages'][-1]['content'].lower()
    assert 'stored-decision' in result['messages'][-1]['content']
    assert next(p for p in result['provenance'] if p['source'] == 'project-file')['path'] == 'a.txt'
    assert result['token_count'] == count(result['messages'])


def test_context_exact_budget_drops_whole_records_never_prompt():
    count = lambda rows: sum(len(r['content']) for r in rows)
    result = select_context('CURRENT', [], project_records=[dict(text='x' * 100)],
                            token_counter=count, budget=7)
    assert result['messages'] == [dict(role='user', content='CURRENT')]
    assert result['omitted'] == 1
    with pytest.raises(ValueError):
        select_context('CURRENT', [], token_counter=count, budget=6)


@pytest.mark.parametrize('flag', ['imported', 'interrupted'])
def test_context_excludes_whole_flagged_pairs(flag):
    rows = [dict(role='user', content='old'), dict(role='assistant', content='answer', **{flag: True})]
    result = select_context('new', rows, token_counter=lambda m: len(str(m)), budget=1000)
    assert result['messages'] == [dict(role='user', content='new')]


def test_provider_overhead_is_counted_and_invalid_counter_rejected():
    with pytest.raises(ValueError):
        select_context('hello', [], token_counter=lambda rows: 10 + len(rows[-1]['content']), budget=14)
    with pytest.raises(ValueError):
        select_context('hello', [], token_counter=lambda rows: True, budget=100)


def test_denied_files_never_opened_by_search(tmp_path, monkeypatch):
    (tmp_path / 'keys.txt').write_text('needle secret')
    (tmp_path / 'safe.md').write_text('needle safe')
    reader = ProjectReader(tmp_path)
    actual = reader.read
    opened = []
    def tracked(path):
        opened.append(path)
        return actual(path)
    monkeypatch.setattr(reader, 'read', tracked)
    assert reader.search('needle')[0]['path'] == 'safe.md'
    assert opened == ['safe.md']


def test_source_secret_is_redacted_with_original_digest(tmp_path):
    raw = 'example credential ghp_' + 'a' * 30
    (tmp_path / 'source.py').write_text(raw)
    reader = ProjectReader(tmp_path)
    record = reader.read('source.py')
    assert record['redacted'] is True
    assert 'ghp_' not in record['text']
    assert record['sha256'] == hashlib.sha256(raw.encode()).hexdigest()
    assert reader.search('example')[0]['redacted'] is True


def test_explicit_root_allowlist_and_context_file_cap(tmp_path):
    approved=tmp_path/'approved';approved.mkdir()
    other=tmp_path/'other';other.mkdir()
    for index in range(4):(approved/f'{index}.txt').write_text('source')
    with pytest.raises(ValueError):ProjectReader(other,allowed_roots=[approved])
    reader=ProjectReader(approved,allowed_roots=[approved],max_files=2)
    assert len(reader.approved_paths())==2
    assert reader.read('0.txt')['untrusted_project_content'] is True
    assert reader.search('source')[0]['untrusted_project_content'] is True


def test_lstat_reparse_attributes_are_rejected_even_in_crossplatform_tests(tmp_path,monkeypatch):
    import stat
    from types import SimpleNamespace
    import neral_control.context as context
    target=tmp_path/'source.txt';target.write_text('safe')
    reader=ProjectReader(tmp_path)
    original=context.os.lstat
    def reparse(path,*args,**kwargs):
        info=original(path,*args,**kwargs)
        if str(path)==str(target):return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_file_attributes=0x400)
        return info
    monkeypatch.setattr(context.os,'lstat',reparse)
    with pytest.raises(ValueError,match='[Rr]eparse|[Ll]ink'):reader.read('source.txt')


@pytest.mark.parametrize('metadata', [
    {'incomplete': True}, {'finish_reason': 'length'}, {'finish_reason': 'cancelled'},
    {'finish_reason': 'timeout'}, {'cancelled': True}, {'timed_out': True},
    {'complete': False}, {'status': 'incomplete'}, {'status': 'failed'},
    {'cancel_requested': True}, {'cancel_acknowledged': True},
    {'complete': 0}, {'complete': 1}, {'complete': None},
    {'incomplete': 'false'}, {'status': []}, {'finish_reason': {}},
])
def test_context_excludes_entire_incomplete_history_pair(metadata):
    rows = [{'role': 'user', 'content': 'old question'},
            {'role': 'assistant', 'content': 'UNFINISHED_SENTINEL', **metadata}]
    result = select_context('current', rows, token_counter=lambda m: len(str(m)), budget=1000)
    assert result['messages'] == [{'role': 'user', 'content': 'current'}]
    assert result['omitted'] == 1


def test_verified_tool_results_survive_before_decisions_files_and_history():
    # A wrong priority order would retain a project file and discard the actual tool result.
    counter = lambda rows: sum(len(row['content']) for row in rows)
    tools = [{'id': 'verified', 'text': 'tool result'}]
    tool_only = select_context('current', [], tool_results=tools,
                               token_counter=counter, budget=10000)
    result = select_context('current', [
        {'role': 'user', 'content': 'old user'}, {'role': 'assistant', 'content': 'old answer'}],
        tool_results=tools, decisions=[{'id': 'decision', 'text': 'old project decision'}],
        project_records=[{'id': 'file', 'text': 'project file'}],
        token_counter=counter, budget=tool_only['token_count'])
    assert result['provenance'] == [{'id': 'verified', 'source': 'tool-result'}]
    assert result['omitted'] == 3
    assert result['messages'][-1]['content'].endswith('current')


def test_project_decisions_survive_before_raw_project_files():
    counter = lambda rows: sum(len(row['content']) for row in rows)
    decisions = [{'id': 'decision', 'text': 'verified project fact'}]
    decision_only = select_context('current', [], decisions=decisions,
                                   token_counter=counter, budget=10000)
    result = select_context('current', [], decisions=decisions,
        project_records=[{'id': 'file', 'text': 'raw project file'}],
        token_counter=counter, budget=decision_only['token_count'])
    assert result['provenance'] == [{'id': 'decision', 'source': 'stored-decision'}]
