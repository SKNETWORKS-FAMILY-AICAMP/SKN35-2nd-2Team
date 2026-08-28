# -*- coding: utf-8 -*-
"""시작 화면 — 리뷰 하나 바로 넣어보기."""
import numpy as np
import pandas as pd
import streamlit as st
# import pydeck as pdk
# from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
from app._shared import get_all, build_row, predict, gauge
from app.theme import apply_theme

apply_theme()
MODEL, GAMES, CARDS, _, META, LANG = get_all()
THR = float(META["임계값"])

st.markdown(
    "<h1 style='text-align:center;margin:8px 0 4px;font-size:30px'>"
    "스팀 게임 리뷰 기반 유저 이탈 예측</h1>"
    "<p style='text-align:center;color:#8f98a0;margin:0 0 24px'>"
    "게임 리뷰를 입력하여 리뷰를 작성한 유저의 이탈 가능성을 예측해보세요</p>",
    unsafe_allow_html=True)

st.write("")

left, right = st.columns([1, 1])
with left:
    # # 순서대로 방문할 나라들 (이름, 위도, 경도)
    # COUNTRIES = [
    #     ("미국", 38.9072, -77.0369),
    #     ("러시아", 55.751667, 37.617778),
    #     ("중국", 35.86166, 104.195397),
    #     ("브라질", -15.7939, -47.8828),
    #     ("스페인", 2.1734035, 41.3850639),
    # ]

    # INTERVAL_MS = 2000  # 이동 주기 (ms)
    # TRAIL_LEN = 4        # 마커 뒤로 남길 잔상 개수
    
    # # ---------------------------------------------------------------------------
    # # 자동 새로고침: INTERVAL_MS 마다 스크립트가 다시 실행됨
    # # ---------------------------------------------------------------------------
    # count = st_autorefresh(interval=INTERVAL_MS, key="map_refresh")
    
    # if "idx" not in st.session_state:
    #     st.session_state.idx = 0
    # else:
    #     st.session_state.idx = count % len(COUNTRIES)
    
    # current_idx = st.session_state.idx
    # current_name, current_lat, current_lon = COUNTRIES[current_idx]
    
    # # st.title("🗺️ 이동하는 마커 지도")
    # # st.caption(f"{INTERVAL_MS/1000:.0f}초마다 다음 나라로 마커가 이동합니다 · 현재: **{current_name}**")
    
    # # ---------------------------------------------------------------------------
    # # 현재 위치 + 최근 방문한 나라들(잔상)을 데이터프레임으로 구성
    # # ---------------------------------------------------------------------------
    # trail_indices = [(current_idx - i) % len(COUNTRIES) for i in range(TRAIL_LEN)]
    # trail_rows = []

    # for rank, idx in enumerate(trail_indices):
    #     name, lat, lon = COUNTRIES[idx]
    #     trail_rows.append(
    #         {
    #             "name": name,
    #             "lat": lat,
    #             "lon": lon,
    #             "is_current": rank == 0,
    #             # 잔상일수록 반투명하게 (RGBA)
    #             "color": [102, 192, 244, 255] if rank == 0 else [102, 192, 244, max(30, 180 - rank * 50)],
    #             "radius": 220000 if rank == 0 else 120000,
    #         }
    #     )
    # df = pd.DataFrame(trail_rows)
    
    # layer = pdk.Layer(
    #     "ScatterplotLayer",
    #     data=df,
    #     get_position=["lon", "lat"],
    #     get_fill_color="color",
    #     get_radius="radius",
    #     pickable=True,
    # )
    
    # view_state = pdk.ViewState(latitude=current_lat, longitude=current_lon, zoom=2.2, pitch=0)
    
    # st.pydeck_chart(
    #     pdk.Deck(
    #         layers=[layer],
    #         initial_view_state=view_state,
    #         map_style="dark",
    #         tooltip={"text": "{name}"},
    #     )
    # )

    COUNTRIES_JS = """
    [
        {name: "미국", lat: 39.8282, lon: -98.5796},
        {name: "러시아", lat: 66.42, lon: 94.25},
        {name: "중국", lat: 35.33,  lon: 103.23},
        {name: "브라질", lat: -14.235, lon: -51.925},
        {name: "스페인", lat: 40.4637, lon: -3.7492}
    ]
    """
    
    HOLD_MS = 2500        # 각 나라에 머무는 시간
    TRANSITION_SEC = 1.6  # 이동(팬) 애니메이션 소요 시간

    html_code = f"""
    <div id="map" style="width:100%; height:560px; border-radius:12px;"></div>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
    const countries = {COUNTRIES_JS};
    let idx = 0;
    
    const map = L.map('map', {{ zoomControl: true }}).setView([countries[0].lat, countries[0].lon], 3);
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
        attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
        maxZoom: 16
    }}).addTo(map);
    
    const marker = L.circleMarker([countries[0].lat, countries[0].lon], {{
        radius: 9, color: '#66c0f4', fillColor: '#66c0f4', fillOpacity: 0.9, weight: 2,
        // 렌더러 클리핑 여유공간을 넉넉히 잡아서, 팬 애니메이션 도중 마커가
        // 화면 경계 밖으로 잠깐 나가도 "사라졌다 나타나는" 현상을 방지
        renderer: L.svg({{ padding: 5 }})
    }}).addTo(map);
    marker.bindTooltip(countries[0].name, {{ permanent: true, direction: 'top', offset: [0, -8] }});
    
    function lerp(a, b, t) {{ return a + (b - a) * t; }}
    
    // 경도는 -180~180 범위를 순환하므로, 그냥 숫자로 보간하면 지구 반대편으로
    // "먼 길"을 도는 경로가 나올 수 있음. 항상 최단 경로(180도 이내)로 보간되도록 보정.
    function lerpLon(fromLon, toLon, t) {{
        let diff = toLon - fromLon;
        if (diff > 180) diff -= 360;
        if (diff < -180) diff += 360;
        let result = fromLon + diff * t;
        if (result > 180) result -= 360;
        if (result < -180) result += 360;
        return result;
    }}
    
    function animateMarker(fromLat, fromLon, toLat, toLon, durationMs) {{
        const start = performance.now();
        function step(now) {{
            const t = Math.min(1, (now - start) / durationMs);
            const lat = lerp(fromLat, toLat, t);
            const lon = lerpLon(fromLon, toLon, t);
            marker.setLatLng([lat, lon]);
            if (t < 1) requestAnimationFrame(step);
        }}
        requestAnimationFrame(step);
    }}
    
    function goToNext() {{
        const from = countries[idx];
        idx = (idx + 1) % countries.length;
        const to = countries[idx];
    
        // 카메라와 마커가 같은 방향(최단 경로)으로 움직이도록, panTo 목적지 경도도
        // from 기준 최단 경로가 되는 값으로 보정 (Leaflet은 -180/180 밖 값도 허용)
        let lonDiff = to.lon - from.lon;
        if (lonDiff > 180) lonDiff -= 360;
        if (lonDiff < -180) lonDiff += 360;
        const adjustedToLon = from.lon + lonDiff;
    
        // 지도(카메라)를 드래그하듯 부드럽게 이동 - panTo는 확대/축소 없이 순수 팬(pan) 애니메이션
        map.panTo([to.lat, adjustedToLon], {{
            animate: true,
            duration: {TRANSITION_SEC},
            easeLinearity: 0.25
        }});
    
        // 마커도 같은 시간 동안 부드럽게 글라이드
        animateMarker(from.lat, from.lon, to.lat, to.lon, {TRANSITION_SEC} * 1000);
    
        marker.unbindTooltip();
        marker.bindTooltip(to.name, {{ permanent: true, direction: 'top', offset: [0, -8] }});
        setTimeout(() => marker.openTooltip(), {TRANSITION_SEC} * 500);
    }}
    
    setInterval(goToNext, {HOLD_MS});
    </script>
    """

    # html_code = f"""
    # <div id="map" style="width:100%; height:560px; border-radius:12px;"></div>
    # <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    # <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    # <script>
    # const countries = {COUNTRIES_JS};
    # let idx = 0;
    # // 카메라의 "실제 누적" 경도 위치. panTo에 준 값을 그대로 계속 이어받아 추적해야
    # // 다음 이동 계산 시 기준이 어긋나지 않음 (-180~180 범위를 벗어난 값을 유지할 수 있음)
    # let cameraLon = countries[0].lon;
    
    # const map = L.map('map', {{ zoomControl: true }}).setView([countries[0].lat, countries[0].lon], 3);
    # L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    #     attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
    #     maxZoom: 16
    # }}).addTo(map);
    
    # const marker = L.circleMarker([countries[0].lat, countries[0].lon], {{
    #     radius: 9, color: '#66c0f4', fillColor: '#66c0f4', fillOpacity: 0.9, weight: 2,
    #     // 렌더러 클리핑 여유공간을 넉넉히 잡아서, 팬 애니메이션 도중 마커가
    #     // 화면 경계 밖으로 잠깐 나가도 "사라졌다 나타나는" 현상을 방지
    #     renderer: L.svg({{ padding: 5 }})
    # }}).addTo(map);
    # marker.bindTooltip(countries[0].name, {{ permanent: true, direction: 'top', offset: [0, -8] }});
    
    # function lerp(a, b, t) {{ return a + (b - a) * t; }}
    
    # // 경도는 -180~180 범위를 순환하므로, 그냥 숫자로 보간하면 지구 반대편으로
    # // "먼 길"을 도는 경로가 나올 수 있음. 항상 최단 경로(180도 이내)로 보간되도록 보정.
    # function lerpLon(fromLon, toLon, t) {{
    #     let diff = toLon - fromLon;
    #     if (diff > 180) diff -= 360;
    #     if (diff < -180) diff += 360;
    #     let result = fromLon + diff * t;
    #     if (result > 180) result -= 360;
    #     if (result < -180) result += 360;
    #     return result;
    # }}
    
    # function animateMarker(fromLat, fromLon, toLat, toLon, durationMs) {{
    #     const start = performance.now();
    #     function step(now) {{
    #         const t = Math.min(1, (now - start) / durationMs);
    #         const lat = lerp(fromLat, toLat, t);
    #         const lon = lerpLon(fromLon, toLon, t);
    #         marker.setLatLng([lat, lon]);
    #         if (t < 1) requestAnimationFrame(step);
    #     }}
    #     requestAnimationFrame(step);
    # }}
    
    # function goToNext() {{
    #     const from = countries[idx];
    #     idx = (idx + 1) % countries.length;
    #     const to = countries[idx];
    
    #     // 카메라의 "실제 누적 위치"(cameraLon) 기준으로 최단 경로를 계산해야 함.
    #     // from.lon(배열 원본, -180~180)을 기준으로 하면 이전 회차에 누적된
    #     // 실제 카메라 위치와 어긋나서, 다음 이동부터 엉뚱한 방향(먼 길)으로 돌게 됨.
    #     let lonDiff = to.lon - cameraLon;
    #     while (lonDiff > 180) lonDiff -= 360;
    #     while (lonDiff < -180) lonDiff += 360;
    #     cameraLon = cameraLon + lonDiff;  // 누적 위치 갱신 (다음 회차 계산의 기준이 됨)
    
    #     // 지도(카메라)를 드래그하듯 부드럽게 이동 - panTo는 확대/축소 없이 순수 팬(pan) 애니메이션
    #     //map.panTo([to.lat, cameraLon], {{
    #     //    animate: true,
    #     //    duration: {TRANSITION_SEC},
    #     //    easeLinearity: 0.25
    #     //}});
    
    #     // 마커는 항상 원본(-180~180) 좌표 기준 최단경로로 이동 - 누적시킬 필요 없음
    #     animateMarker(from.lat, from.lon, to.lat, to.lon, {TRANSITION_SEC} * 1000);
    
    #     marker.unbindTooltip();
    #     marker.bindTooltip(to.name, {{ permanent: true, direction: 'top', offset: [0, -8] }});
    #     setTimeout(() => marker.openTooltip(), {TRANSITION_SEC} * 500);
    # }}
    
    # setInterval(goToNext, {HOLD_MS});
    # </script>
    # """

    components.html(html_code, height=580)
    
    st.markdown("---")
    st.caption(
        "구현 방식: Leaflet.js를 브라우저에 직접 임베드하고 setInterval로 순환. "
        "카메라는 map.panTo(...)로 순수 팬(드래그) 애니메이션, 마커는 requestAnimationFrame으로 "
        "선형 보간(lerp)해 같은 시간 동안 함께 이동시켰습니다. Streamlit 재실행과 무관하게 "
        "브라우저 안에서 계속 스스로 동작합니다."
    )

with right:
    st.markdown("""
        <div style='text-align: center; font-size: 40px'>It's GOOOOOOOOOOOOOOOOOD!!!</div>
    """, unsafe_allow_html=True)
        
st.write("")
left, center, right = st.columns([5, 2, 5])
with center:
    if st.button("▶  PLAY", width="stretch"):
        st.switch_page("pages/1_작별인사_판별기.py")
