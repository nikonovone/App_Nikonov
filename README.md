# Поля, которые будут использоваться для поиска вакансий соискателями

- **vacancy_custom_position**
    
        Название вакансии - первое, что видит соискатель, по нему же определяют переходить ли к подробностм вакансии

- **vacancy_operating_schedule_id**
        
        Для многих соискателей подходят только конкретные графики работ
- **vacancy_salary_from**

        Минимальная зарплата один из наиболее важных показателей у вакансии
- **vacancy_description**

        Описание вакансии необходимо для понимания требований к работнику и условий работы.
- **vacany_company_id**

        Зачастую соискатели ищут работы в конкретной компании, или же, когда встречают знакомое название, то увеличивается интерес к вакансии
- **vacancy_city_id**

        Большая часть соискателей смотрит вакансии в первую очередь в родном городе
- **vacancy_offer_experience_year_count**

        Поможет соискателю сопоставить свой опыт с требуемыми компетенциями
- **vacancy_offer_education_id**

        Поможет соискателю настроить более гибкий фильтр по вакансиям, в зависимости от его образования

# План эксперимента по расчету релевантности выдачи поисковой системы.
1. Цель эксперимента - расчет релевантности выдачи поисковой системы.

        Предполагается, что она должна поменятся после каких-либо изменений в системе поиска.
2. Входные параметры - достаточное для устранения стат. погрешности количество поисковых запросов.          
Выходные параметры - ранжированные ответы от поисковой системы.

        Предполагается, что данные на которых будет проверяться система, чистые и верно размеченные, например ассесорами.
3. Выбор метрики для расчёта релевантности поисковых выдач системы.

        В зависимости от внешних условий необходимо выбрать наиболее подходящую метрику из доступных, будь то классические метрики, метрики на основе ранговой корреляции или метрики на основе каскадной модели поведения
4. Тестирование методом переплетения на реальных данных.

        В случае если метрика увеличилсь на расчетное, необходимое число, можно попробовать потестировать систему методом переплетения на реальных пользователях.

5. Интерпретация результатов эксперимента