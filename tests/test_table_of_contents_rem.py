#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from text_normalize import normalize_text, replacements_regex


class TableOfContentsRemovalTest(unittest.TestCase):

    # ЕЮ Устав.docx
    def test_remove_1(self):
        t = '''
УТВЕРЖДЕНО



Решением общего собрания акционеров

акционерного общества «Евротэк - Югра»



(Протокол № [_] от [_]  2019 г.)

___________________________________________________________________

УСТАВ

АКЦИОНЕРНОГО ОБЩЕСТВА

«ЕВРОТЭК-ЮГРА»

___________________________________________________________________



Российская Федерация

г. Москва

2019 год


СОДЕРЖАНИЕ

Статья                                                                                                                                                              Стр.

1.	Общие положения	2

2.	Цель и предмет деятельности Общества	3

3.	Уставный капитал Общества. Акции, облигации и другие эмиссионные ценные бумаги. Вклады в имущество Общества	4

4.	Передача акций. Обременение акций, преимущественное право на приобретение акций. Согласие на передачу акций	6

5.	Права и обязанности Акционеров	8

6.	Органы управления Общества	9

7.	Общее собрание акционеров	9

8.	Совет Директоров	17

9.	Единоличный исполнительный орган Общества	25

10.	Внутренняя Ревизионная комиссия и Аудитор	27

11.	Спорные вопросы	27

12.	Бухгалтерский учет и отчетность	28

13.	Распределение прибыли и дивиденды	28

14.	Резервный фонд и другие фонды	28

15.	Документы и информация	29

16.	Разрешение споров	29

17.	Определения	30






Общие положения 

	Предмет регулирования

        '''

        normal_text = normalize_text(t, replacements_regex)
        print(normal_text)
        print('='*20)

        self.assertTrue(len(normal_text)<len(t)*0.9)

    # МНГ Устав.docx
    def test_remove_2(self):
        t = '''
У  С  Т  А В

Акционерного общества

«Мессояханефтегаз»

(новая редакция)



г. Новый Уренгой 2017 г.



ОГЛАВЛЕНИЕ

	Статья 1. Общие положения	3

	Статья 2. Фирменное наименование и место нахождения Общества	3

	Статья 3. Юридический статус Общества	3

	Статья 4. Ответственность Общества	4

	Статья 5. Филиалы и представительства,  дочерние и  зависимые общества	4

	Статья 6. Цель создания и основные виды деятельности Общества	5

Статья 7. Уставный капитал Общества. Размещенные и объявленные акции. Вклады в имущество общества, не увеличивающие уставный

	капитал	6

	Статья 8. Облигации и иные эмиссионные ценные бумаги Общества	7

	Статья 9. Права и обязанности акционеров Общества	7

	Статья 10. Фонды Общества	8

	Статья 11. Дивиденды Общества	8

	Статья 12. Органы управления	9

	Статья 13. Общее собрание акционеров Общества	9

	Статья 14. Компетенция Общего собрания акционеров Общества	10

	Статья 15. Решения Общего собрания акционеров Общества	11

Статья 16. Предложения в повестку дня, информация о проведении

	Общего собрания акционеров Общества	12

Статья 17. Участие и голосование на Общем собрании акционеров,

	протокол Общего собрания акционеров Общества	13

	Статья 18. Совет директоров Общества	14

	Статья 19. Компетенция Совета директоров Общества	14

	Статья 20. Председатель Совета директоров, Секретарь заседания Совета директоров и Секретарь Совета директоров Общества	20

	Статья 21. Заседания Совета директоров Общества	21

	Статья 22. Генеральный директор Общества	22

	Статья 23. Порядок отчуждения акций Общества. Ограничения в отношении сделок с акциями Общества	26

	Статья 24. Ревизионная комиссия Общества	27

	Статья 25. Действие Устава Общества	29



Статья 1. Общие положения



	Акционерное общество «Мессояханефтегаз» (прежние фирменные наименования: Открытое акционерное общество «Мессояханефтегаз», Закрытое акционерное общество «Мессояханефтегаз»), именуемое в дальнейшем «Общество», создано с наименованием Открытое акционерное общество «Мессояханефтегаз» путем реорганизации в форме выделения из ЗАО «Заполярнефтегазгеология» и зарегистрировано распоряжением Главы Тазовского района Ямало-Ненецкого автономного округа Тюменской области (свидетельство о государственной регистрации от 13 августа 1998 г. № 273).

	Правовое положение Общества, права и обязанности его акционеров определяются Гражданским кодексом Российской Федерации, Федеральным законом «Об акционерных обществах», другими правовыми актами Российской Федерации, настоящим Уставом, а также корпоративным договором в отношении Общества.
        '''

        normal_text = normalize_text(t, replacements_regex)
        print(normal_text)
        print('=' * 20)

        self.assertTrue(len(normal_text) < len(t) * 0.9)

    # 6.1.1(a) Project Tri-Neft - Sunrise Charter.docx
    def test_remove_3(self):
        t = '''
УСТАВ

CHARTER

ОБЩЕСТВА С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ «ГАЗПРОМНЕФТЬ-ВОСТОК»

OF LIMITED LIABILITY COMPANY GAZPROMNEFT-VOSTOK

(редакция № [])

(version No. [])





Томск

Tomsk 

2018 год

2018



СОДЕРЖАНИЕ

TABLE OF CONTENTS

1.	ОБЩИЕ ПОЛОЖЕНИЯ	4

2.	ПРАВОВОЙ СТАТУС ОБЩЕСТВА	7

3.	ОТВЕТСТВЕННОСТЬ ОБЩЕСТВА	8

4.	ЦЕЛИ И ВИДЫ ДЕЯТЕЛЬНОСТИ ОБЩЕСТВА	8

5.	Уставный капитал Общества	9

6.	УВЕЛИЧЕНИЕ УСТАВНОГО КАПИТАЛА ОБЩЕСТВА ЗА СЧЕТ ДОПОЛНИТЕЛЬНЫХ ВКЛАДОВ УЧАСТНИКОВ	10

7.	ПРАВА И ОБЯЗАННОСТИ УЧАСТНИКОВ	12

8.	ВЫХОД УЧАСТНИКА ИЗ ОБЩЕСТВА	16

9.	ПЕРЕДАЧА ДОЛИ УЧАСТНИКА	16

10.	ПРЕИМУЩЕСТВЕННОЕ ПРАВО	19

11.	ЗАЛОГ ИЛИ ИНОЕ ОБРЕМЕНЕНИЕ ДОЛИ	27

12.	СПИСОК УЧАСТНИКОВ ОБЩЕСТВА	27

13.	ИМУЩЕСТВО ОБЩЕСТВА	28

14.	РАСПРЕДЕЛЕНИЕ ПРИБЫЛИ	29

15.	УПРАВЛЕНИЕ ОБЩЕСТВОМ	31

16.	ОБЩЕЕ СОБРАНИЕ УЧастников	31

17.	КОМПЕТЕНЦИЯ ОБЩЕГО СОБРАНИЯ УЧАСТНИКОВ	32

18.	КВОРУМ И ПОРОГОВЫЕ ЗНАЧЕНИЯ ДЛЯ ГОЛОСОВАНИЯ НА ОБЩЕМ СОБРАНИИ УЧАСТНИКОВ	35

19.	ПРОЦЕДУРА СОЗЫВА ОБЩЕГО СОБРАНИЯ УЧАСТНИКОВ	39

20.	ПРОЦЕДУРА УЧАСТИЯ В ОБЩЕМ СОБРАНИИ УЧАСТНИКОВ	41

21.	ПРОЦЕДУРА ПРОВЕДЕНИЯ ОБЩЕГО СОБРАНИЯ УЧАСТНИКОВ	42

22.	СОВЕТ ДИРЕКТОРОВ ОБЩЕСТВА	44

23.	ПРОЦЕДУРА ПРОВЕДЕНИЯ ЗАСЕДАНИЙ СОВЕТА ДИРЕКТОРОВ	60

24.	ЕДИНОЛИЧНЫЙ ИСПОЛНИТЕЛЬНЫЙ ОРГАН ОБЩЕСТВА И УПРАВЛЕНИЕ ОБЩЕСТВОМ	63

25.	СДЕЛКИ СО СВЯЗАННЫМИ СТОРОНАМИ	68

26.	АУДИТОР	69

27.	ХРАНЕНИЕ ДОКУМЕНТОВ ОБЩЕСТВА	70

28.	РЕОРГАНИЗАЦИЯ И ЛИКВИДАЦИЯ ОБЩЕСТВА	70

29.	ОПРЕДЕЛЕНИЯ	71

30.	ПРОЧИЕ ПОЛОЖЕНИЯ	82



1.	GENERAL PROVISIONS	4

2.	LEGAL STATUS OF THE COMPANY	7

3.	LIABILITY OF THE COMPANY	8

4.	PURPOSES AND BUSINESS ACTIVITIES OF THE COMPANY	8

5.	charter CAPITAL OF THE COMPANY	9

6.	increase of the charter CAPITAL OF THE COMPANY BY ADDITIONAL CONTRIBUTIONS OF THE PARTICIPANTS	10

7.	RIGHTS AND OBLIGATIONS OF THE Participants	12

8.	WITHDRAWAL OF THE PARTICIPANT FROM THE COMPANY	16

9.	TRANSFER OF A Participant’S PARTICIPATION INTEREST	16

10.	PRE-EMPTIVE RIGHT	19

11.	PLEDGE OR OTHER ENCUMBRANCE OF A PARTICIPATION INTEREST	27

12.	LIST OF THE COMPANY’S ParticipantS	27

13.	ASSETS OF THE COMPANY	28

14.	DISTRIBUTION OF PROFIT	29

15.	MANAGEMENT OF THE COMPANY	31

16.	GENERAL MEETING OF THE ParticipantS	31

17.	COMPETENCE OF THE GENERAL MEETING OF THE ParticipantS	32

18.	QUORUM AND VOTING PASSMARK OF THE GENERAL MEETING OF THE PARTICIPANTS	35

19.	PROCEDURE FOR CONVENING THE GENERAL MEETING OF THE PARTICIPANTS	39

20.	PROCEDURE FOR PARTICIPATION IN GENERAL MEETING OF THE PARTICIPANTS	41

21.	PROCEDURE FOR CONDUCTING THE GENERAL MEETING OF THE PARTICIPANTS	42

22.	BOARD OF DIRECTORS OF THE COMPANY	44

23.	PROCEDURE FOR CONDUCTING MEETINGS OF THE BOARD	60

24.	SOLE EXECUTIVE BODY OF THE COMPANY AND MANAGEMENT	63

25.	TRANSACTIONS WITH RELATED PARTIES	68

26.	AUDITOR	69

27.	CUSTODY OF DOCUMENTS OF THE COMPANY	70

28.	REORGANIZATION AND LIQUIDATION OF THE COMPANY	70

29.	DEFINITIONS	71

30.	MISCELLANEOUS	82







ОБЩИЕ ПОЛОЖЕНИЯ

Общество с ограниченной ответственностью «Газпромнефть-Восток», далее именуемое «Общество», было первоначально учреждено в качестве Общества с ограниченной ответственностью «Сибнефть-Восток» на основании решения об учреждении от 31 августа 2005 года (Свидетельство о государственной регистрации от 14 сентября 2005 года, серия 70 № 000990939, выдано Налоговой инспекцией Федеральной налоговой службы по г. Томску).

Информация об Обществе была внесена в Единый государственный реестр юридических лиц под Основным государственным регистрационным номером 1057002610378 Налоговой инспекцией Федеральной налоговой службы по г. Томску 14 сентября 2005 года.
        '''

        normal_text = normalize_text(t, replacements_regex)
        print(normal_text)
        print('=' * 20)

        self.assertTrue(len(normal_text) < len(t) * 0.9)

    # Договор_ООО Зодчий_25 млн.$.docx
    def test_remove_4(self):
        t = '''
г. Тюмень

2019г.






Оглавление

		1.	ПРЕДМЕТ ДОГОВОРА	3

		2.	ТЕРМИНЫ И ОПРЕДЕЛЕНИЯ	3

		3.	ВЗАИМООТНОШЕНИЯ СТОРОН	5

		4.	ПРАВА И ОБЯЗАННОСТИ СТОРОН	5

		4.1.	Права и обязанности Заказчика:	5

		4.2.	Права и обязанности Исполнителя:	10

		4.3.	Управление эффективностью деятельности контрагентов	17

		5.	ПЕРСОНАЛ ИСПОЛНИТЕЛЯ	19

		6.	ПОРЯДОК ОКАЗАНИЯ УСЛУГ	20

		7.	СТОИМОСТЬ УСЛУГ, ПОРЯДОК ИХ ПРИЕМКИ И РАСЧЕТОВ. МЕХАНИЗМ ИЗМЕНЕНИЯ ОБЪЕМА УСЛУГ ПО ДОГОВОРУ.	22

		8.	ОТВЕТСТВЕННОСТЬ СТОРОН	28

		9.	ПРОИЗВОДСТВО РАБОТ. ТРЕБОВАНИЯ В ОБЛАСТИ  ОХРАНЫ ТРУДА, ПРОМЫШЛЕННОЙ БЕЗОПАСНОСТИ И ОХРАНЫ ОКРУЖАЮЩЕЙ СРЕДЫ (ПЭБ, ОТ и ГЗ) ПРИ ОКАЗАНИИ УСЛУГ НА ОБЪЕКТАХ ЗАКАЗЧИКА	31

		10.	ОБСТОЯТЕЛЬСТВА НЕПРЕОДОЛИМОЙ СИЛЫ. САНКЦИИ.	32

		11.	ПОРЯДОК ИЗМЕНЕНИЯ, РАСТОРЖЕНИЯ ДОГОВОРА	33

		12.	ПРОЧИЕ УСЛОВИЯ	36

		13	РАССМОТРЕНИЕ СПОРОВ И АРБИТРАЖ	37

		14.	КОНФИДЕНЦИАЛЬНОСТЬ	38

		15.	СРОК ДЕЙСТВИЯ ДОГОВОРА	39

		16.	ПРИЛОЖЕНИЯ К ДОГОВОРУ	40

		17.	ЮРИДИЧЕСКИЕ, ПОЧТОВЫЕ АДРЕСА, БАНКОВСКИЕ И ПЛАТЕЖНЫЕ РЕКВИЗИТЫ СТОРОН	40




ДОГОВОР №_______

на оказание услуг по телеметрическому и технологическому сопровождению

при наклонн
        '''

        normal_text = normalize_text(t, replacements_regex)
        print(normal_text)
        print('=' * 20)

        self.assertTrue(len(normal_text) < len(t) * 0.9)

