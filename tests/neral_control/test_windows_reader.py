"""Win32 traversal contracts on Linux doubles, plus native tests on Windows.

The doubles prove decisions/flags, not real NTFS/junction race protection.
A native Windows run and adversarial rename/junction tests remain release gates.
"""
import ctypes
import ntpath
import os
from types import SimpleNamespace
import pytest
from neral_control.context import _read_windows, _WindowsAPI, ProjectReader

GUID='\\\\?\\Volume{12345678-1234-1234-1234-123456789abc}\\'

class Handles:
 def __init__(self,*,bad_kind=None,reparse=None,escape=None,volume_change=None,size=3):
  self.opened=[];self.closed=[];self.reads=[];self.bad_kind=bad_kind;self.reparse=reparse;self.escape=escape;self.volume_change=volume_change;self.size=size
 def open(self,path,*,directory):
  n=len(self.opened);self.opened.append((path,directory))
  assert not self.closed,'ancestors must remain pinned until read completes'
  return n
 def info(self,h):
  directory=self.opened[h][1]
  return ((0x10 if directory else 0)|(0x400 if h==self.reparse else 0),self.size,2 if h==self.volume_change else 1,2 if h==self.bad_kind else 1)
 def final_path(self,h):
  if h==self.escape:return GUID+'outside\\secret.txt'
  return GUID if h==0 else self.opened[h][0]
 def read(self,h,limit):self.reads.append((h,limit));return b'abc'
 def close(self,h):self.closed.append(h)
 def list(self,path):return [('a.txt',False)]


def test_windows_pins_every_ancestor_then_reads_canonical_volume_path():
 api=Handles()
 assert _read_windows(r'C:\approved\project',['src','a.py'],8192,api)==b'abc'
 assert api.opened==[('C:'+chr(92),True),(GUID+'approved',True),(GUID+'approved\\project',True),(GUID+'approved\\project\\src',True),(GUID+'approved\\project\\src\\a.py',False)]
 assert api.reads==[(4,8193)] and api.closed==[4,3,2,1,0]

@pytest.mark.parametrize('issue',['reparse','escape','volume_change','bad_kind'])
def test_windows_rejects_reparse_junction_escape_volume_device_before_read(issue):
 api=Handles(**{issue:2})
 with pytest.raises(ValueError):_read_windows(r'C:\approved\project',['a.py'],8192,api)
 assert api.reads==[] and api.closed==list(range(len(api.opened)-1,-1,-1))


def test_windows_oversize_and_failed_child_open_release_all_parent_handles():
 api=Handles(size=8193)
 with pytest.raises(ValueError):_read_windows(r'C:\approved',['a.py'],8192,api)
 assert api.reads==[] and api.closed==[2,1,0]
 api=Handles();actual=api.open
 def broken(path,*,directory):
  if path.endswith('a.py'):raise PermissionError('sharing violation')
  return actual(path,directory=directory)
 api.open=broken
 with pytest.raises(PermissionError):_read_windows(r'C:\approved',['a.py'],8192,api)
 assert api.closed==[1,0] and api.reads==[]

@pytest.mark.parametrize('path',[r'\\server\share\project',r'\\?\C:\project',r'C:relative',r'C:\project.'])
def test_windows_ambiguous_or_network_roots_rejected(path):
 api=Handles()
 with pytest.raises(ValueError):_read_windows(path,['a.py'],8192,api)
 assert api.reads==[]


def test_win32_open_flags_deny_write_and_delete_sharing():
 calls=[]
 api=_WindowsAPI.__new__(_WindowsAPI)
 api.ctypes=SimpleNamespace(c_void_p=ctypes.c_void_p)
 api.dll=SimpleNamespace(CreateFileW=lambda *args:(calls.append(args) or 17))
 assert api.open('C:\\',directory=True)==17
 assert api.open('C:\\a.py',directory=False)==17
 for args in calls:
  assert args[2]==1 # FILE_SHARE_READ only, no WRITE or DELETE sharing.
  assert args[4]==3 # OPEN_EXISTING, never create/write.
  assert args[5] & 0x00200000 and args[5] & 0x02000000
 assert calls[0][1]==0x80 and calls[1][1]==0x80000000


def test_windows_enumeration_occurs_while_directory_handles_are_pinned():
 api=Handles()
 def listing(path):
  assert not api.closed
  assert path==GUID+'approved\\project'
  return [('a.py',False)]
 api.list=listing
 assert _read_windows(r'C:\approved\project',[],8192,api,directory_only=True)==[('a.py',False)]
 assert api.closed==[2,1,0]


@pytest.mark.skipif(os.name!='nt',reason='Native Win32/NTFS tests require a real Windows executor')
def test_native_windows_regular_file_and_symlink_refusal(tmp_path):
 (tmp_path/'safe.py').write_text('print(42)',encoding='utf8')
 reader=ProjectReader(tmp_path)
 assert reader.read('safe.py')['text']=='print(42)'
 try:(tmp_path/'linked.py').symlink_to(tmp_path/'safe.py')
 except OSError:pytest.skip('Windows developer mode/symlink privilege required')
 with pytest.raises(ValueError):reader.read('linked.py')
