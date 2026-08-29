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

COUNTRIES_JS = """
    [
        {name: "미국", lat: 39.8282, lon: -98.5796, review: "Gosh, I love when a big black guy that is thrice the size of me can run faster than Octane on a stim then bum me since I deal 10% less damage. I only play this game because I can relate to Wraith and Mirage. Do not play this game if you are pregnant because youll get kicked in the stomach by a three stack of controllers."},
        {name: "러시아", lat: 66.42, lon: 94.25, review: "Сюжет детский, как будто для десятилетних писали. НПС безжизненные, стоят на месте и повторяют одни и те же фразы, никакого влияния на мир игры не чувствуется. Хогвартс красивый, атмосфера есть, но история и персонажи — слабое место."},
        {name: "중국", lat: 35.33,  lon: 103.23, review: "凭借对HP的热爱, 最终吃下了整个罐头, 获得了美味的全成就. 作为一个HP八年的原著粉来说, 打开游戏看到霍格沃茨时, 就值了. 主线真的很短😭战斗开困难模式也很简单, 由于我不会捏脸，顶着赫敏·波特的脸推完…美术真的很好!"},
        {name: "브라질", lat: -14.235, lon: -51.925, review: "Por muito tempo foi o melhor Battle Royale, mas pra mim se tornou um Simulador de FOMO, são mapas demais, personagens sendo lançados com uma frequência muito alta. Se tornou impossível ter uma boa experiência sem se dedicar APENAS em Apex."},
        {name: "스페인", lat: 40.4637, lon: -3.7492, review: "juego de ♥♥♥♥♥♥ , lo llevo jugando desde que salió y no ha visto una season sin bugs de server ,visuales o de audio, un asco revivan el titanfall, han pasado 2 semanas de esta reseña y confirmo , cada vez peor la ♥♥♥♥♥♥ de juego."}
    ]
"""

HOLD_MS = 10000     # 각 나라에 머무는 시간
TRANSITION_SEC = 3  # 이동(팬) 애니메이션 소요 시간

html_code = f"""
    <div style="display: flex; width: 100%">
        <div id="map" style="width:50%; height:500px; border-radius:12px;"></div>
        
        <div id="review-panel" style="
            padding:16px 20px; border-radius:10px; min-height:70px; width: 50%;
            font-family:-apple-system,'Segoe UI',Roboto,sans-serif; color:#e6f0f9;
        ">
            
            <div style="font-size:30px; color:#8f98a0; margin-bottom:6px;">
                🎮 <span id="review-country">-</span>
            </div>
            
            <div style="font-size:30px; line-height:1.5;">
                <span id="review-text"></span><span id="type-cursor">▍</span>
            </div>
        </div>
    </div>
    
    <style>
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
        
        function typeReview(countryName, text) {{
            const nameEl = document.getElementById('review-country');
            const textEl = document.getElementById('review-text');
            const cursorEl = document.getElementById('type-cursor');
        
            if (typingTimer) clearTimeout(typingTimer);  // 스트리밍 도중 다음 나라로 넘어간 경우 이전 타이머 정리
            nameEl.textContent = countryName;
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

                    typeReview(to.name, to.review)
                }}
            );
        }}

        // 최초 로딩 시 첫 번째 나라의 리뷰도 바로 타이핑
        typeReview(countries[0].name, countries[0].review);

        setInterval(goToNext, {HOLD_MS});
    </script>
"""

st.iframe(html_code, height=580)