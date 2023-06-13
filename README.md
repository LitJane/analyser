 

# Analyser core & nlp_tools

## CLI  
см. [основные команды для работы с анализатором](cli/README.md)

___

## Конве́йер анализатора
0. Пользователь предоставляет документы. 

2. Модуль [Parser](https://github.com/nemoware/document-parser) получает на вход документы Word и преобразует их в удобное внутреннее представление.
Его результаты попадают в Mongo DB. Документы ставятся в очередь на обработку.
   
2. Модуль анализатора вынимает из очереди документ. Тект разбивается на токены.

3. Текст нормализуется: убирается "мусор" -- непечатаемые символы, двойные пробелы, лишние дефисы, лишние переносы, т.п..  
   Регистр слов нормализуется на основе статистики (например, известные аббревиатуры переводятся в верхний регистр и т. д.).
   
4. Производится эмбеддинг (встраивание?) текста -- перевод токенов в векторное представление. 
   Для "встраивания" (русский, пощади!) мы используем предварительно обученную DeepPavlov модель ELMO.
   
5. Формируется _матрица признаков токенов_ -- каждому токену 
   присваивается соотв. бинарный вектор, единицы в котором 
   указывают на наличие того или иного признака, например, 
   «лексема знака препинания», или «содержит только цифры», 
   или «начинается с прописной буквы» и т. д.
   
6. _Матрица эмбеддинга_ и _матрица признаков токенов_ подается на вход нейронной сети.
   
7. На выходе из сети -- 
   - _семантическая карта текста_ -- бинарная матрица, высота которой кратна количесву обнаружеваемых атрибутов, а длинна соотв. кол-ву токенов в тексте.
Единицами в строках матрицы помечены (подсвечены) зоны текста, соотв. атрибутам.
   - one-hot вектор, характеризующий предмет договора (классификация)
   
8. По _семантической карте текста_ определяются координаты 
   найденных признаков. Найденные значения атрибутов очищаются и нормализуются --
     например, кавычки вокруг названий компаний отрезаны,
     имена преобразовываются в стандартный регистр и т. д.
   
9. Найденные атрибуты записываются в виде дерева (Бодхи) в Mongodb 
10. Выявленные атрибуты документов используются далее на этапе Аудита и подготовки заключения.
   
## Некоторые подходы к обучению нейронной модели (в общих чертах)
### TL
- Мы полагаемся на предварительно обученные открытые модели третьих сторон, 
    применяются механизмы Transfer learning (TL).
### MTL
- Мы используем multi-task learning (MTL) подход к обучению, -- одни и те же веса 
    распределяются между классификацией предмета контракта и определением атрибутов.
### Обучающий набор   
3. Мы ограниченно применяем так называемый протокол обучения Noisy Student -- 
    обучающий набор содержит небольшое количество документов, автоматически размеченных
    предыдущей версией модели, в то время как большинство документов размечены биологическим интеллектом 
   ~~безнадежно устаревших~~ людей.
   
### Расширение обучающего набора (augmentation)
- Размер входного тензора ограничен до 600. 
  Мы случайным образом вырезаем из документа фрагмент длиной 600 токенов, что, по сути, 
  дает возможность сделать из одного обучающего примера много подобных.
   
- К эмбеддингам  добавляется некоторое кол-во шума.


### Miscl   
- На некотором этапе обучения модель может сваливаться в локальные 
минимумы и плохо распознавать определенный признак,
    в то время как другая версия обученной модели может этот признак распозновать хорошо.
    В таком случае мы производим линейное взвешивание двух набораов весов, чтобы получить 
«комбинированный» набор отягощений, с которых продолжается тренировка.

#### Длинные тексты   
- Поскольку некоторые документы слишком длинные, чтобы сразу поместиться во входной слой модели,
    мы анализируем их, применяя "движущееся окно" с перекрытием 20%. Перекрывающиеся края окон
    линейно взвешены.
  

#### Балансировка обучающего набора  
- Балансировка обучающего набора достигается за счет присвоения весов обучающим примерам. 
  Веса пропорциональны редкости предмета договора, качеству разметки и тп.
  
- Для редких документов создается больше augmented версий.


  

## Обучающий набор
TODO: 🚧

- Обучающий набор в основе является множеством структур json и содержится в Mongo DB.

   Структура несет: 
   - Токенизированный текст, а именно -- координаты всех токенов. 
     Минимальной единицей является слово 
      
   - Атрибуты, которые ссылаются на координаты токенов -- на начало и 
     конец последовательности. Атрибуты выстроены в древовидную структуру.

- Для обучения формируются пары X1, X2 : Y1, Y2,
  где 
  - X1 -- Матрица эмбеддингов (600 x 1024);
  - X2 -- _Матрица признаков токенов_ (600 x ?) (см выше)
  - Y1 -- _семантическая карта текста_ --  (600 x ?)
  - Y2 -- one-hot вектор, характеризующий предмет договора (классификация)


## Подготовка обучающего набора
- Первичная (грязная) обучающая выборка создавалась 
    в полуавтоматическом режиме, для чего использовались 
    различные эвристики, нечеткий поиск шаблонов, регулярные выражения.
Небольшое количество документов было промаркировано/исправлено 
вручную.
На основе этого была создана первая модель, которая позже 
использовалась для разметки следующего поколения 
обучающей выборки. 🚧
Такой итеративный подход (Noisy Student) позволял работать 
при отсутствии достаточного количества обучающих примеров.
    
- TODO: 🚧  

## Архитектура нейронной модели

TODO: 🚧  

![Архитектура нейронной модели, схема](analyser/vocab/model_att.png)

## Структура директорий

- analyser
 -- основной код
- analyser/vocab: данные -- словари, вспомогательные модели и т.д.
- bin : утилиты командной строки, связанные с выводом (запуск служб и т. д.).
- [cli](cli) : утилиты командной строки, связанные с обучением, для запуска 
  обучения, публикации модели и т. д.
- experiments: Jupyter Notebooks для тестирования различных идей, связанных с обучением.
- colab_support: утилиты для отображения данных в Jupyter Notebooks
- gpn: константы, связвнные с организацией
- integration: интерфейсы для интеграции со сторонними системами
- tests: юнит-тесты 
- tf_support: фрейморк-зависимые утилиты для расширения для TensorFlow.
- training_reports: отчеты о процессе обучения, структуре обучающей выборки и т.д.
- trainsets: Jupyter Notebook для обучения и формирования отчетов 
- work: кэши и проч. промежуточные файлы, необходимые в ран-тайм фазе 
___

## Miscl. commands
- Create wheel: 
```
python setup.py bdist_wheel 
```
- Collect all wheels of the project: 
```
pip wheel -r requirements.txt --wheel-dir=tmp/wheelhouse
```
- Install collected wheels 
``` 
pip install --no-index --find-links=tmp/wheelhouse SomePackage 
```

## Assign a release tag:
1. Create a tag:
    ```                     
    > git tag -a vX.X.X -m "<release comment>"
    ```
1. Push tag:
    ```                     
    > git push origin --tags
    ```

## Usage (Windows):
1. Install Python >=3.6 and pip
1. Install ```virtualenv```( [https://virtualenv.pypa.io/en/latest/installation/]() ):
    ```
    > pip install virtualenv
    ```
1. Download ``` nemoware_analyzer-X.X.X-py3-none-any.whl ``` to working dir (e.g. ```analyser_home```)     
1. Change to the work dir:
    ```
    > cd analyser_home
    ```
1. Create virtual environment (with name ```venv```):
    ```
    > virtualenv venv
    ```
1. Activate:
    ``` 
    > .\venv\Scripts\activate
    ```
1. Install ```analyser``` with all deps:
    ```
    > pip install  .\nemoware_analyzer-X.X.X-py3-none-any.whl    
    ```
1. Run:

    ```
    > analyser_run
    ```


## Run analyzer as a service
1. Register systemd service
    ```
    > cd bin 
    > sudo ./install_service.sh 
    ```
1. Service commands
    ```
    sudo systemctl stop nemoware-analyzer.service          #To stop running service 
    sudo systemctl start nemoware-analyzer.service         #To start running service 
    sudo systemctl restart nemoware-analyzer.service       #To restart running service 
    ```
    
# CML 
### contintinous machine learning
CML is triggered only on push or pull request to `model` branch.  
refer https://github.com/nemoware/analyser/.github/workflows/cml.yaml

to run CML worker (just example, parameters may differ):

```
sudo docker run --name <ANYNAME> -d 
   -v ~/pip_cache:/pip_cache 
   -v ~/gpn:/gpn_cml 
   -e GPN_WORK_DIR=/gpn_cml   
   -e RUNNER_IDLE_TIMEOUT=18000  
   -e RUNNER_LABELS=cml,cpu   
   -e RUNNER_REPO=https://github.com/nemoware/analyser   
   -e repo_token=<personal github access token> 
   -e GPN_DB_HOST=192.168.10.36 
   -e PIP_DOWNLOAD_CACHE=/pip_cache    
   dvcorg/cml-py3
```

