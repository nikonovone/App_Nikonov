# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# Загрузка необходимых библиотек
import pandas as pd
import numpy as np
import plotly.express as px

from dash import Dash, dcc, html
from utils import *
from functools import partialmethod
from scipy.stats import mstats

# Настройки к Pandas
pd.DataFrame.head = partialmethod(pd.DataFrame.head, n=2)
pd.options.display.max_rows = 13
pd.options.display.max_columns = 20

app = Dash(__name__)
server = app.server
# Чтение датасета
df = pd.read_csv("https://media.githubusercontent.com/media/nikonovone/App_Nikonov/master/data.csv", index_col=0)
#df = pd.read_csv("data.csv", index_col=0)

font = dict(
    family="Gotham Pro Medium",
    size=12,
)


# Визуализируем Топ-10 вакансий
data = df['vacancy_custom_position'].value_counts()[:10]
top10 = px.bar(df, x=data.index, y=data, color_discrete_sequence=['#6189c9'])
top10.update_xaxes(tickangle=0)
top10.update_layout(
    xaxis=dict(
        tickfont=dict(size=6)))
top10.update_layout(
    title='Топ 10 самых популярных вакансий',
    xaxis_title="Название вакансий",
    yaxis_title="Количество, шт",
    font=font
)

# Мин. зарплата
df['vacancy_salary_from'] = df['vacancy_salary_from'].replace(-1, 0)
data = pd.Series(mstats.winsorize(
    df[df['vacancy_salary_from'] != 0]['vacancy_salary_from'], limits=[0, 0.05]))
min_salary = px.histogram(df, x=data, marginal="box", color_discrete_sequence=['#6189c9'])
min_salary.update_layout(
    title='Распределение минимальной зарплаты в 95% вакансий',
    xaxis_title="Зарплата, руб",
    yaxis_title="Вакансии, шт",
    font=font,
)

# Макс. зарплата
df['vacancy_salary_to'] = df['vacancy_salary_to'].replace(-1, np.nan)
data = pd.Series(mstats.winsorize(
    df['vacancy_salary_to'].dropna(), limits=[0, 0.05]))
max_salary = px.histogram(x=data, marginal="box",  color_discrete_sequence=['#6189c9'])
max_salary.update_layout(
    title='Распределение максимальной зарплаты в 95% вакансий',
    xaxis_title="Зарплата, руб",
    yaxis_title="Количество, шт",
    font=font
)
# Образование
data = df['vacancy_offer_education_id']
education_cat = px.histogram(x=data, marginal="box", color_discrete_sequence=['#6189c9'])
education_cat.update_layout(
    title='Распределение категорий образования в вакансиях',
    xaxis_title="Категории",
    yaxis_title="Вакансии, шт",
    font=font
)

y = pd.Series(mstats.winsorize(df['vacancy_salary_from'], limits=[0, 0.05]))
education_bar = px.box(y=y, x=df['vacancy_offer_education_id'], color_discrete_sequence=['#6189c9'])
education_bar.update_layout(
    title='Распределение минимальной зарплаты в зависимости от категории образования',
    xaxis_title="Категория образования",
    yaxis_title="Зарплата, руб",
    font=font
)
# Опыт работы
df['vacancy_offer_experience_year_count'].replace(
    [-1, -100], np.nan, inplace=True)
data = df['vacancy_offer_experience_year_count']
offer_experience = px.histogram(x=data, marginal="box",  color_discrete_sequence=['#6189c9'])
offer_experience.update_layout(
    title='Распределение требуемого опыта работы в вакансиях',
    yaxis_title="Вакансии, шт",
    xaxis=dict(
        tickmode='linear',
        title="Опыт работы, лет",
    ),
    font=font
)

offer_experience_bar = px.box(y=y, x=df['vacancy_offer_experience_year_count'], color_discrete_sequence=['#6189c9'])
offer_experience_bar.update_layout(
    title='Распределение минимальной зарплаты в зависимости от опыта работы',
    xaxis_title="Категория образования",
    yaxis_title="Зарплата, руб",
    font=font
)
# График работы
shedule = px.box(y=y, x=df['vacancy_operating_schedule_id'], color_discrete_sequence=['#6189c9'])
shedule.update_layout(
    title='Распределение минимальной зарплаты в зависимости от графика работы',
    xaxis_title="График работы",
    yaxis_title="Зарплата, руб",
    font=font
)
# Города
data = pd.Series(mstats.winsorize(df['vacancy_city_id'], limits=[0, 0.05]))
cities = px.histogram(x=data, marginal="box",  color_discrete_sequence=['#6189c9'])
cities.update_layout(
    title='Распределение id городов в 95% вакансий',
    xaxis_title="ID города",
    yaxis_title="Вакансии, шт",
    font=font
)

# Корреляция числовых признаков
corrs = df.drop(columns=['vacany_company_id', 'vacancy_is_agency',
                         'vacancy_offer_experience_year_id', 'vacancy_city_id']).corr()
correlations =  px.imshow(np.array(np.round(corrs,2)),
                text_auto=True,
                aspect="auto",
                color_continuous_scale=px.colors.sequential.Cividis,
                x=['Зарплата от', 'Зарплата до', 'Опыт работы', 'Диапазон зарплаты', ' '],
                y=['Зарплата от', 'Зарплата до', 'Опыт работы', 'Диапазон зарплаты',' ']
                )
correlations.update_layout(
    title='Карта корреляций числовых признаков',
    font=font
)


app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="🐬", className="header-emoji"),
                html.H1(
                    children="Bimbo аналитика",
                    className="header-title"
                ),
                html.P(
                    children="Анализируем тестовые датасеты с 1995 года",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.H1(children='Основные распределения данных'),
                html.Div(
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=top10,
                        ),
                        html.Div( 
                            children=[
                                dcc.Markdown('''
                                        Самые популярные профессии - простые,
                                        не требующие высоких компетенций у соискателя
                                ''')
                            ],
                            className='conclusion'
                        )
                    ],
                    className="card",
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=shedule,
                        ),
                        html.Div( 
                            children=[
                                dcc.Markdown('''
                                        Как видно в сравнение 7 категории с остальными - от графика
                                        работы существенно меняется медианная зарплата
                                        Отметим, что 1 и 3 категории графика работы составляют **70%** всех вакансий в датасете
                                ''')
                            ],
                            className='conclusion'
                        )
                    ],
                    className="card",
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=min_salary,
                        ),
                        html.Div( 
                            children=[
                                dcc.Markdown('''
                                        Стоит отметить привязку пиков к круглым суммам -
                                        30, 35, 40, 45 тысяч рублей и т.д.
                                        Медианной минимальной зарплатой в 95% вакансий
                                        является **40 000** рублей

                                ''')
                            ],
                            className='conclusion'
                        )
                    ],
                    className="card",
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=max_salary,
                        ),
                        html.Div( 
                            children=[
                                dcc.Markdown('''
                                        Медианной максимальной зарплатой в 95% вакансий
                                        является **55 000** рублей

                                ''')
                            ],
                            className='conclusion'
                        )
                    ],
                    className="card",
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=education_bar,
                        ),
                        html.Div( 
                            children=[
                                dcc.Markdown('''
                                        В данном датасете образование не оказывает
                                        значительное влияние на зарплату

                                ''')
                            ],
                            className='conclusion'
                        )
                    ],
                    className="card",
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=education_cat,
                        ),
                        html.Div( 
                            children=[
                                dcc.Markdown('''
                                       В большинстве вакансий требуется образование
                                    категории "0"

                                ''')
                            ],
                            className='conclusion'
                        )
                    ],
                    className="card",
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=offer_experience_bar,
                        ),
                        html.Div( 
                            children=[
                                dcc.Markdown('''
                                        Можно выделить линейную зависимость от 1 до 4 лет опыта
                                        работы, потом наблюдается плато по изменению зарплаты,
                                        вероятно такой график можно объяснить искаженными
                                        и не очищенными данными

                                ''')
                            ],
                            className='conclusion'
                        )
                    ],
                    className="card",
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=offer_experience,
                        ),
                        html.Div( 
                            children=[
                                dcc.Markdown('''
                                        Медианный опыт работы требуемый в вакансиях -
                                        **1** год. Также заметен пик на значении 3 года, что
                                        может объясняться спросом на опытных специалистов

                                ''')
                            ],
                            className='conclusion'
                        )
                    ],
                    className="card",
                ),
                
                html.Div(
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=cities,
                        ),
                        html.Div( 
                            children=[
                                dcc.Markdown('''
                                        На графике видно несколько пиков, вероятно это
                                        **Москва**, **Санкт-Петербур**г и некоторые города 
                                        миллионники, в которых сосредоточена большая часть 
                                        вакансий. Сумма 1, 2, 57, 4, 35 категорий городов имеет **69%**
                                        от общего количества вакансий.

                                ''')
                            ],
                            className='conclusion'
                        )
                    ],
                    className="card",
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=correlations,
                        ),
                        html.Div( 
                            children=[
                                dcc.Markdown('''
                                        Явных корреляционных зависимостей нет,
                                        кроме зарплаты от и до, что логично.

                                ''')
                            ],
                            className='conclusion'
                        )
                    ],
                    className="card",
                )
            ],
            className="graphs",
        ),
    ]
)




if __name__ == '__main__':
    app.run_server(debug=True)
