import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyparsing import empty
import plotly.express as px
import plotly.graph_objects as go

def transform(x):
    if x <= 1:
        return 1 + (1 - x)
    else:
        return 1 - (x - 1)


st.set_page_config(layout="wide")

# DATA READ
df_team = pd.read_csv("lol_data/lol_team_data.csv");
df_player = pd.read_csv("lol_data/lol_players_data.csv");

# 팀 설정
with st.sidebar:
    team = st.selectbox('팀 선택:', df_team['Name'].unique())

# lis = 선택한 팀의 선수 데이터(df)
lis=df_player[df_player['team']==team]

# fi_columns = 선수 이름, 팀, 국적 제외 데이터(df): 스테이터스
fi_columns = lis.columns.difference(['Player', 'team', 'Country', 'Positon'])
# 데이터 전처리
# lis['XPD_15']=lis['XPD_15']/100
# lis['GPM']=lis['GPM']/100
# lis['DPM']=lis['DPM']/100
# lis['GD_15']=lis['GD_15']/100

# 첫 번째 행
empty2,con1,team_chart = st.columns([1.0,1.0,0.6])
# 그래프 표시되는 부분
col1_1, col1_2, col1_3 = st.columns([3.5,1.0,1.0])
col2_1, col2_2, col2_3 = st.columns([3.5,1.0,1.0])

# 첫 번째 행 설정
with team_chart:
    
    plt.rcParams['font.family'] = 'Malgun Gothic'

    # 데이터 전처리 및 설정
    score_Kill = (df_team[df_team.Name == team]['Kills&game'].mean() / df_team['Kills&game'].mean()).round(2)
    score_Death = transform(df_team[df_team.Name == team]['Deaths&game'].mean() / df_team['Deaths&game'].mean()).round(2)
    score_Gold = (df_team[df_team.Name == team]['GPM'].mean() / df_team['GPM'].mean()).round(2)
    score_Damage = (df_team[df_team.Name == team]['DPM'].mean() / df_team['DPM'].mean()).round(2)
    score_Vision = (
        (df_team[df_team.Name == team]['WPM'].mean() / df_team['WPM'].mean()).round(2) +
        (df_team[df_team.Name == team]['VWPM'].mean() / df_team['VWPM'].mean()).round(2) * 2 +
        (df_team[df_team.Name == team]['WCPM'].mean() / df_team['WCPM'].mean()).round(2)
    ) / 4

    categories = ('킬', '데스', '골드', '데미지', '시야', '킬')
    score = (score_Kill, score_Death, score_Gold, score_Damage, score_Vision, score_Kill)
    all_team_score = (1, 1, 1, 1, 1, 1)
    # 각도 계산
    theta = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)

    # 그래프 그리기
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi = 300, figsize = (4, 6))
    fig.patch.set_facecolor('none')  # 배경 색상 설정

    ax.plot(theta, score, label=team, 
            color = 'darkorange', lw = 2)
    ax.fill_between(theta, 0, score, color='orange', alpha=0.7)
    ax.plot(theta, all_team_score, label='평균치',
            color = 'gray', lw = 2)

    # 그래프 설정
    plt.ylim(0, 1.7)

    # 라벨 붙이기
    ax.set_xticks(theta)
    ax.set_xticklabels(categories, color = 'white', fontsize = 25, y=-0.2)
    ax.set_rticks([x for x in np.arange(0.0, 1.7, 0.4)])
    ax.set_yticklabels(['' for x in np.arange(0., 1.7, 0.4)])

    ax.set_theta_zero_location('N')

    # plt.legend()
    
    st.pyplot(fig)

with con1:
    src='lol_data/img/'+team+'.png'
    img = Image.open(src)
    img = img.resize((350, 180))
    st.image(img)

with empty2 :
    empty()

# 간격 미리 설정
# with empty1_3 :
#     empty()

# with empty1_4 :
#     empty()

# with empty2_3 :
#     empty()

# with empty2_4 :
#     empty()

# 2행 1열 -> 팀별 통계 그래프
with col1_1:
    # 선수들 킬과 데스 그래프
    data = df_player[df_player.team == team][['Player', 'Avg_kills', 'Avg_deaths']]

    fig = go.Figure()
    # 킬 데이터
    fig.add_trace(go.Bar(x=data['Player'], y=data['Avg_kills'], name='Kills',
                         marker_color = 'orange',
                         width=0.7, text=data['Avg_kills']))
    # 데스 데이터
    fig.add_trace(go.Bar(x = data['Player'], y = data['Avg_deaths'] * -1,
                         name='Deaths', marker_color = 'red',
                         width = 0.7, text=data['Avg_deaths']))
    # 최고 포인트, 최저 포인트 등 표시
    fig.add_shape(type='line',
              x0=-0.5, x1=4.5, y0=df_player['Avg_kills'].mean(), y1=df_player['Avg_kills'].mean(),
              line=dict(color='red', width=2, dash='dash'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='red', size=10),
                             name='Mean Kill Score'))
    
    fig.add_shape(type='line',
              x0=-0.5, x1=4.5, y0=df_player['Avg_deaths'].mean() * -1, y1=df_player['Avg_deaths'].mean() * -1,
              line=dict(color='yellow', width=2, dash='dash'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='yellow', size=10),
                             name='Mean Death Score'))
    
    fig.update_layout(barmode='overlay',
                      title='Kill - Death',
                      xaxis_title='Players',
                      yaxis_title='Scores',
                      legend=dict(title='Score Type'))

    
    st.plotly_chart(fig)

# 2열 2행 -> 골드 및 골드 차이 그래프
with col1_2:
    data = df_player[df_player.team == team][['Player', 'GPM', 'GD_15']]

    fig = go.Figure()
    
    # 분당 골드 데이터
    fig.add_trace(go.Bar(x=data['Player'], y=data['GPM'], name='Gold Per Min',
                         marker_color = 'orange',
                         width=0.7, text=data['GPM']))

    # 분당 골드 차이
    fig.add_trace(go.Scatter(x = data['Player'], y = data['GD_15'], name = 'Gold Difference at 15 min',
                             mode = 'markers',
                             marker=dict(color='red', size=18,
                                         line = dict(width = 4, color = 'white')),
                             text = data['GD_15']
                             ))
    
    # 레이아웃 설정
    fig.update_layout(title='About Gold',
                      xaxis_title='Players',
                      yaxis_title='Gold',
                      legend=dict(title='Score Type'))
    st.plotly_chart(fig)

# 3열 1행 -> 와드와 시야
with col2_1:
    data = df_player[df_player.team == team][['Player', 'Avg_WPM', 'Avg_WCPM', 'Avg_VWPM', 'VSPM']]
    
    fig = go.Figure()
    
    # 와드에 관한 데이터
    for i, player in enumerate(data['Player']):
        fig.add_trace(go.Scatter(x=[data['Avg_WPM'].iloc[i] + data['Avg_VWPM'].iloc[i]], 
                                y=[data['Avg_WCPM'].iloc[i]],
                                name=f'{player} | VS: {data["VSPM"].iloc[i]}',
                                mode='markers',
                                marker=dict(color='white', size=data['VSPM'].iloc[i] * 25,
                                            line=dict(width=4, color=['orange', 'red', 'blue', 'green', 'yellow'][i])),
                                text=player,
                                textposition='middle center',
                                legendgroup=player  # 각 선수를 별도의 범례 그룹으로 설정
                                ))
    
    fig.update_layout(title='Vision',
                      xaxis_title='Wards Place Per Min',
                      yaxis_title='Destroy Wards Per Min',
                      legend=dict(title='Score Type'))
    
    st.plotly_chart(fig)
    
    
# 3열 2행 -> 데미지에 관련된 차트
with col2_2:
    data = df_player[df_player.team == team][['Player', 'DMG(%)']]
    subdata = [df_player[df_player.Position == df_player.Position[i]]['DMG(%)'].mean().round(2) for i in range(5)]
    
    fig = go.Figure()
    # 데미지 비율에 관한 데이터
    fig.add_trace(go.Bar(x=np.arange(5) - 0.18, y=data['DMG(%)'], name='Damage(%)',
                         marker_color = 'orange',
                         width=0.35, text=data['DMG(%)']))
    fig.add_trace(go.Bar(x=np.arange(5) + 0.18, y=subdata, name='All_PLayer_Damage(%)',
                         marker_color = 'lightgreen',
                         width=0.35, text=subdata))
    
    fig.update_layout(title='Damage(%)',
                      xaxis_title='Player',
                      yaxis_title='Damage Persent',
                      legend=dict(title='Score Type'))
    
    fig.update_xaxes(tickvals=np.arange(len(data['Player'])), ticktext=data['Player'])

    
    st.plotly_chart(fig)