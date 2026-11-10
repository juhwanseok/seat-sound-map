"""우리금융아트홀 공연 집계 (r02 할 일 #6)
출처: 올림픽공원 누리집 공연/행사 일정안내(ksponco.or.kr/olympicpark/eventInfo/eventInfoList?mid=a20301010100)가
화면을 그릴 때 부르는 공개 목록(eventInfoListAjax.es). 공공데이터포털 3037877(올림픽공원 공연정보, 행사명·행사기간·행사장소)과
같은 기관의 같은 일정 정보다. 인증키 없이 공개 화면과 똑같은 요청을 월별 1회씩만 보낸다.
- 공연장 코드 plcCd=04 → 우리금융아트홀
- 산출: out/events_woori_<연도>.json (공연 목록, 공연일 수, 제목 기준 장르 분류, 좌석 공급량 하한)
사용: python -I scripts/events_woori.py 2025
"""
import sys, io, json, os, re, time, datetime as dt
import urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
URL = 'https://www.ksponco.or.kr/olympicpark/eventInfo/eventInfoListAjax.es'
SEATS = 1184   # 공식 좌석 수(1층 832·2층 352, 기관 누리집)
RAW = f'raw/events/{YEAR}'; os.makedirs(RAW, exist_ok=True)

ev = {}
for m in range(1, 13):
    body = json.dumps({'nowYM': f'{YEAR}{m:02d}', 'plcCd': '04'}).encode()
    req = urllib.request.Request(URL, data=body, method='POST', headers={
        'Content-Type': 'application/json;charset=UTF-8', 'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://www.ksponco.or.kr/olympicpark/eventInfo/eventInfoList?mid=a20301010100'})
    res = json.loads(urllib.request.urlopen(req, timeout=30).read().decode('utf-8'))
    json.dump(res, open(f'{RAW}/{m:02d}.json', 'w', encoding='utf-8'), ensure_ascii=False)
    for part in res:
        for it in part:
            ev[it['evtSeq']] = {k: it.get(k) for k in ('evtSeq', 'title', 'plc', 'beginDt', 'endDt', 'tm')}
    time.sleep(1.0)

def d(s): return dt.date.fromisoformat(s.strip()[:10])
y0, y1 = dt.date(YEAR, 1, 1), dt.date(YEAR, 12, 31)
days = set(); rows = []
for e in ev.values():
    if '우리금융' not in (e['plc'] or ''):
        continue
    b, f = max(d(e['beginDt']), y0), min(d(e['endDt']), y1)
    if b > f:
        continue
    n = (f - b).days + 1
    span = {b + dt.timedelta(k) for k in range(n)}
    days |= span
    t = e['title']
    genre = ('뮤지컬' if re.search(r'뮤지컬|musical', t, re.I) else
             '콘서트·음악' if re.search(r'콘서트|concert|라이브|live|음악회|리사이틀|오케스트라|어쿠스틱|재즈|팬미팅', t, re.I) else
             '연극·낭독' if re.search(r'연극|낭독|play', t, re.I) else '기타')
    rows.append({**e, 'days_in_year': n, 'genre_by_title': genre})
rows.sort(key=lambda r: r['beginDt'])
by_genre = {}
for r in rows:
    g = by_genre.setdefault(r['genre_by_title'], {'n': 0, 'days': 0}); g['n'] += 1; g['days'] += r['days_in_year']
out = {'year': YEAR, 'source': URL + ' (plcCd=04, 월별 nowYM)', 'fetched_at': dt.datetime.now().isoformat(timespec='seconds'),
       'n_events': len(rows), 'event_days_union': len(days), 'seats_per_show': SEATS,
       'seat_supply_lower_bound': len(days) * SEATS,
       'note': '공연일 수는 공연 기간을 합친 날 수(휴관일 포함 가능). 하루 1회 공연으로 본 좌석 공급량 하한이며 실제 회차·관객 수가 아니다. 장르는 제목 키워드 분류.',
       'by_genre_title': by_genre, 'events': rows}
os.makedirs('out', exist_ok=True)
json.dump(out, open(f'out/events_woori_{YEAR}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ('year', 'n_events', 'event_days_union', 'seat_supply_lower_bound', 'by_genre_title')}, ensure_ascii=False, indent=1))
for r in rows: print(f" {r['beginDt']}~{r['endDt']}  {r['days_in_year']:3d}일  [{r['genre_by_title']}] {r['title'][:50]}")
