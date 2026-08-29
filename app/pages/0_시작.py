# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from app._shared import get_all, build_row, predict, gauge
from app.theme import apply_theme

apply_theme()
MODEL, GAMES, CARDS, _, META, LANG = get_all()
THR = float(META["임계값"])

st.markdown(
    "<h1 style='text-align:center; font-size:40px'>"
    "스팀 게임 리뷰 기반 유저 이탈 예측</h1>"
    "<div style='text-align: center; color: #8f98a0; margin:0 0px 24px'>"
    "게임 리뷰를 입력하여 리뷰를 작성한 유저의 이탈 가능성을 예측해보세요.</div>",
    unsafe_allow_html=True)

left, center, right = st.columns([5, 2, 5])
with center:
    if st.button("▶  PLAY", width="stretch"):
        st.switch_page("pages/1_작별인사_판별기.py")

# PLAY 버튼과 지도가 붙어 있으면 답답하다. 한 칸 띄우고 소제목을 준다.
st.markdown(
    '<div style="height:38px"></div>'
    '<div class="tourhead">세계에서 올라온 리뷰</div>'
    '<div class="toursub">국기는 리뷰를 쓴 <b>언어</b>의 대표 지역이며 작성자의 국적이 아닙니다.</div>',
    unsafe_allow_html=True)

# ★ review 본문은 실제 스팀에서 가져온 것이다. 그대로 둔다.
#   나머지(up · hrs · posted · helpful · user · products)는
#   **스팀 카드 모양을 보여주려고 채운 예시 값**이다. 우리 수집 데이터가 아니다.
#   발표에서 이 숫자를 근거로 말하면 안 된다.
COUNTRIES_JS = """
    [
        {name: "미국", flag: "US", lat: 39.8282, lon: -98.5796,
         up: false, hrs: "412.6", posted: "23 August", helpful: 11, awards: 0,
         user: "ProfanityPenguin", products: 251,
         review: "Gosh, I love when a big black guy that is thrice the size of me can run faster than Octane on a stim then bum me since I deal 10% less damage. I only play this game because I can relate to Wraith and Mirage. Do not play this game if you are pregnant because youll get kicked in the stomach by a three stack of controllers."},
        {name: "러시아", flag: "RU", lat: 66.42, lon: 94.25,
         up: false, hrs: "58.3", posted: "19 August", helpful: 34, awards: 1,
         user: "Kirill_74", products: 63,
         review: "Сюжет детский, как будто для десятилетних писали. НПС безжизненные, стоят на месте и повторяют одни и те же фразы, никакого влияния на мир игры не чувствуется. Хогвартс красивый, атмосфера есть, но история и персонажи — слабое место."},
        {name: "중국", flag: "CN", lat: 35.33,  lon: 103.23,
         up: true, hrs: "126.9", posted: "21 August", helpful: 87, awards: 3,
         user: "XiaoHei", products: 342,
         review: "凭借对HP的热爱, 最终吃下了整个罐头, 获得了美味的全成就. 作为一个HP八年的原著粉来说, 打开游戏看到霍格沃茨时, 就值了. 主线真的很短😭战斗开困难模式也很简单, 由于我不会捏脸，顶着赫敏·波特的脸推完…美术真的很好!"},
        {name: "브라질", flag: "BR", lat: -14.235, lon: -51.925,
         up: false, hrs: "1,204.8", posted: "17 August", helpful: 52, awards: 2,
         user: "lucasBR", products: 88,
         review: "Por muito tempo foi o melhor Battle Royale, mas pra mim se tornou um Simulador de FOMO, são mapas demais, personagens sendo lançados com uma frequência muito alta. Se tornou impossível ter uma boa experiência sem se dedicar APENAS em Apex."},
        {name: "스페인", flag: "ES", lat: 40.4637, lon: -3.7492,
         up: false, hrs: "0.9", posted: "25 August", helpful: 6, awards: 0,
         user: "elmiguel", products: 17,
         review: "juego de ♥♥♥♥♥♥ , lo llevo jugando desde que salió y no ha visto una season sin bugs de server ,visuales o de audio, un asco revivan el titanfall, han pasado 2 semanas de esta reseña y confirmo , cada vez peor la ♥♥♥♥♥♥ de juego."}
    ]
"""

HOLD_MS = 10000     # 각 나라에 머무는 시간
TRANSITION_SEC = 3  # 이동(팬) 애니메이션 소요 시간

html_code = f"""
    <div style="display: flex; width: 100%">
        <div id="map" style="width:50%; height:500px; border-radius:12px;"></div>
        
        <div id="review-panel" style="width:50%; padding-left:18px;">
            <!-- 스팀 스토어의 리뷰 카드 구조를 그대로 흉내낸다.
                 위에 '도움이 됐다' 줄, 그 아래 추천 배지 + 기록상 시간,
                 작성일, 본문, 맨 아래 작성자. -->
            <div class="sr-card">
                <div class="sr-top">
                    <span id="sr-helpful">0 people found this review helpful</span>
                    <span class="sr-award">🏅 <b id="sr-awards">0</b></span>
                </div>

                <div class="sr-head">
                    <div class="sr-thumb" id="sr-thumb">👎</div>
                    <div>
                        <div class="sr-verdict" id="sr-verdict">Not Recommended</div>
                        <div class="sr-hrs"><b id="sr-hrs">0.0</b> hrs on record</div>
                    </div>
                    <img class="sr-flag" id="sr-flag" alt=""
                             title="리뷰를 쓴 언어의 대표 지역 (작성자 국적 아님)">
                </div>

                <div class="sr-posted">Posted: <span id="sr-posted">-</span></div>

                <div class="sr-body">
                    <span id="review-text"></span><span id="type-cursor">▍</span>
                </div>

                <div class="sr-foot">
                    <div class="sr-avatar" id="sr-avatar">?</div>
                    <div>
                        <div class="sr-user" id="sr-user">-</div>
                        <div class="sr-products"><span id="sr-products">0</span> products in account</div>
                    </div>
                    <div class="sr-country" id="review-country">-</div>
                </div>
            </div>
        </div>
    </div>
    
    <style>
    /* ── 스팀 리뷰 카드 ─────────────────────────────────────── */
    .sr-card {{
        background:#1b2838; border:1px solid #2a475e; border-radius:10px;
        padding:14px 18px 0; height:500px; box-sizing:border-box;
        display:flex; flex-direction:column;
        font-family:-apple-system,'Segoe UI',Roboto,sans-serif; color:#c7d5e0;
    }}
    .sr-top {{
        display:flex; align-items:center; gap:10px;
        font-size:12.5px; color:#8f98a0;
        border-bottom:1px solid #2a475e; padding-bottom:10px;
    }}
    .sr-award {{ margin-left:auto; color:#66c0f4; }}
    .sr-head {{ display:flex; align-items:center; gap:13px; padding:14px 0 4px; }}
    .sr-thumb {{
        width:46px; height:46px; border-radius:4px; display:grid; place-items:center;
        font-size:24px; background:rgba(224,92,92,.16); flex:0 0 46px;
    }}
    .sr-thumb.up {{ background:rgba(102,192,244,.16); }}
    .sr-verdict {{ font-size:19px; font-weight:700; color:#e05c5c; line-height:1.2; }}
    .sr-verdict.up {{ color:#66c0f4; }}
    .sr-hrs {{ font-size:13px; color:#8f98a0; margin-top:2px; }}
    .sr-hrs b {{ color:#c7d5e0; font-weight:600; }}
    .sr-flag {{ margin-left:auto; width:42px; height:auto; border-radius:3px;
               border:1px solid #2a475e; display:block; }}
    .sr-posted {{ font-size:12.5px; color:#8f98a0; padding:8px 0 12px; }}
    .sr-body {{
        font-size:16px; line-height:1.65; color:#dce6ef;
        flex:1; overflow:auto; padding-bottom:12px;
    }}
    .sr-foot {{
        display:flex; align-items:center; gap:11px;
        border-top:1px solid #2a475e; padding:11px 0 13px; margin-top:auto;
    }}
    .sr-avatar {{
        width:34px; height:34px; border-radius:3px; flex:0 0 34px;
        background:linear-gradient(150deg,#2a475e,#1b2838); border:1px solid #2a475e;
        display:grid; place-items:center; font-size:15px; color:#8f98a0;
    }}
    .sr-user {{ font-size:13.5px; font-weight:600; color:#66c0f4; }}
    .sr-products {{ font-size:11.5px; color:#8f98a0; }}
    .sr-country {{ margin-left:auto; font-size:12.5px; color:#8f98a0; }}

    #type-cursor {{
        display: inline-block;
        color: #66c0f4;
        font-size: 20px;
        margin-left: 3px;
        vertical-align: middle;
        animation: pulse-cursor 1s ease-in-out infinite;
    }}
    @keyframes pulse-cursor {{
        0%, 100% {{ opacity: 0.25; transform: scale(0.85); }}
        50% {{ opacity: 1; transform: scale(1); }}
    }}
    #type-cursor.done {{ display: none; }}
    .fade-chunk {{
        opacity: 0;
        display: inline;
        animation: fade-in-chunk 0.4s ease forwards;
    }}
    @keyframes fade-in-chunk {{
        from {{ opacity: 0; transform: translateY(3px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>

    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const countries = {COUNTRIES_JS};
        let idx = 0;

        // 비행기 이모지 아이콘 정의
        const planeEmojiIcon = L.divIcon({{
            html: '<div style="font-size: 28px; line-height: 1; transform: rotate(0deg);">✈️</div>',
            className: 'custom-plane-icon', // 기본 Leaflet 테두리 스타일 제거용
            iconSize: [30, 30],
            iconAnchor: [15, 15] // 아이콘의 중심점 좌표 (크기의 절반)
        }});

        const map = L.map('map', {{ zoomControl: true }}).setView([countries[0].lat, countries[0].lon], 3);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
            maxZoom: 16
        }}).addTo(map);

        const marker = L.marker([countries[0].lat, countries[0].lon], {{
            radius: 9, color: '#66c0f4', fillColor: '#66c0f4', fillOpacity: 0.9, weight: 2,
            renderer: L.svg({{ padding: 5 }}),
            icon: planeEmojiIcon
        }}).addTo(map);

        // ---- 답변 스트리밍 애니메이션 ----
        let typingTimer = null;
        
        function typeReview(c) {{
            const textEl = document.getElementById('review-text');
            const cursorEl = document.getElementById('type-cursor');
            const text = c.review;

            if (typingTimer) clearTimeout(typingTimer);  // 이전 타이머 정리

            // 스팀 카드 채우기
            document.getElementById('review-country').textContent = c.name;
            document.getElementById('sr-flag').src =
                'https://flagcdn.com/w80/' + (c.flag || 'un').toLowerCase() + '.png';
            document.getElementById('sr-helpful').textContent =
                (c.helpful || 0) + ' people found this review helpful';
            document.getElementById('sr-awards').textContent = c.awards || 0;
            document.getElementById('sr-hrs').textContent = c.hrs;
            document.getElementById('sr-posted').textContent = c.posted;
            document.getElementById('sr-user').textContent = c.user;
            document.getElementById('sr-products').textContent = c.products;
            document.getElementById('sr-avatar').textContent =
                (c.user || '?').slice(0, 1).toUpperCase();

            const thumb = document.getElementById('sr-thumb');
            const verdict = document.getElementById('sr-verdict');
            thumb.textContent = c.up ? '\\u{{1F44D}}' : '\\u{{1F44E}}';
            thumb.classList.toggle('up', !!c.up);
            verdict.textContent = c.up ? 'Recommended' : 'Not Recommended';
            verdict.classList.toggle('up', !!c.up);

            textEl.innerHTML = '';
            cursorEl.classList.remove('done');
        
            // 공백을 유지한 채 단어 단위로 쪼갬 (예: ["이 ", "지역 ", "유저들은 "])
            const chunks = text.match(/\\S+\\s*/g) || [text];
            let i = 0;
        
            function appendNext() {{
                if (i >= chunks.length) {{
                    cursorEl.classList.add('done');  // 다 나오면 커서 숨김
                    typingTimer = null;
                    return;
                }}
                const span = document.createElement('span');
                span.textContent = chunks[i];
                span.className = 'fade-chunk';
                textEl.appendChild(span);
        
                const lastChar = chunks[i].trim().slice(-1);
                i += 1;
        
                // 기본 간격(55~125ms)에 무작위성을 줘서 일정하지 않게, 문장부호 뒤엔 더 쉬어가게
                // let delay = 55 + Math.random() * 70;
                let delay = 100;
                if (['.', ',', '!', '?'].includes(lastChar)) delay += 180;
        
                typingTimer = setTimeout(appendNext, delay);
            }}
            appendNext();
        }}

        marker.bindTooltip(countries[0].name, {{ permanent: true, direction: 'top', offset: [0, -20] }});

        function lerp(a, b, t) {{ return a + (b - a) * t; }}

        function animateMarker(fromLat, fromLon, toLat, toLon, durationMs, onComplete) {{
            const start = performance.now();

            function step(now) {{
                const t = Math.min(1, (now - start) / durationMs);
                const lat = lerp(fromLat, toLat, t);

                // 복잡한 lerpLon 대신 단순 lerp 사용 (날짜변경선을 가로지르지 않음)
                const lon = lerp(fromLon, toLon, t);
                marker.setLatLng([lat, lon]);
                if (t < 1) {{
                    requestAnimationFrame(step);
                }} else {{
                    if (onComplete) onComplete();
                }}
            }}

            requestAnimationFrame(step);
        }}

        function goToNext() {{
            const from = countries[idx];
            idx = (idx + 1) % countries.length;
            const to = countries[idx];

            marker.unbindTooltip();

            // 최단거리 보정(lonDiff) 없이 목적지 좌표(to.lon)로 직접 이동
            map.panTo([to.lat, to.lon], {{
                animate: true,
                duration: {TRANSITION_SEC},
                easeLinearity: 0.75
            }});

            // 마커도 동일하게 원본 경도 사이를 단순 보간하여 이동
            animateMarker(from.lat, from.lon, to.lat, to.lon, {TRANSITION_SEC} * 1000, 
                () => {{
                    marker.bindTooltip(to.name, {{ 
                        permanent: true, 
                        direction: 'top', 
                        offset: [0, -20] 
                    }});

                    typeReview(to)
                }}
            );
        }}

        // 최초 로딩 시 첫 번째 나라의 리뷰도 바로 타이핑
        typeReview(countries[0]);

        setInterval(goToNext, {HOLD_MS});
    </script>
"""

st.iframe(html_code, height=580)