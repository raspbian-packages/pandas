#!/usr/bin/python3

# testing methods of finding the architecture, for e.g. #973584

import sys
import sysconfig
import platform
import struct

print("973584 test - uname:",platform.uname(),"sysp:",sys.platform,"getplatform:",sysconfig.get_platform(),"pp:",platform.platform(),"parch:",platform.architecture(),"byteorder:",sys.byteorder,"maxsize:",sys.maxsize,"psize",struct.calcsize("P"))
