from pat16.imaging import anatomy, motion_correct
from pat16 import SESSIONS
import argparse
p=argparse.ArgumentParser();p.add_argument('--session',choices=SESSIONS);p.add_argument('--kind',choices=['dwi','func']);a=p.parse_args()
anatomy()
for session in ([a.session] if a.session else SESSIONS):
    for kind in ([a.kind] if a.kind else ['dwi','func']): motion_correct(kind,session)
