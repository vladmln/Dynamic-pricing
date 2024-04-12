# Динамическое ценообразование для торговых площадок с использованием методов машинного обучения

## Описание проекта

Проект направлен на разработку системы динамического ценообразования товаров на торговых площадках, используя методы машинного обучения. Модель предсказывает оптимальную цену продажи товара, основываясь на исторических данных о продажах, включая информацию о ценах, продажах, спецификации товаров и временных интервалах. Целью является максимизация прибыли за счет оптимизации цен в реальном времени, учитывая спрос и предложение, а также другие рыночные факторы.

## Рабочее окружение

Модель разработана с использованием Python и библиотек для машинного обучения: PyTorch, NumPy, Pandas, Scikit-learn.

## Установка зависимостей

Для установки необходимых библиотек используйте команду:
pip install numpy pandas torch sklearn

## Подготовка данных

Данные должны быть предварительно обработаны и включать следующие основные атрибуты: дата продажи, идентификатор магазина, идентификатор товара, цену товара, количество проданных единиц, а также дополнительные атрибуты, которые могут повлиять на спрос, такие как месяц, год и день продаж.
### В данный момент данные очищаются от продаж по слишком низкой цене (квантиль 0.01) и от слишком больших количеств продаж (0.99 квантиль)
### В данный момент используются следующие фичи:
- `item_id` - id товара
- `shop_id` - id магазина
- `item_category_id` - нормализованный id категории товара (id / сумму всех id)
- `item_price` - цена товара
- `month` - месяц
- `year` - год
- `day` - день

## Структура проекта

- `data_preprocessing.py` - скрипт для предобработки данных.
- `model.py` - описание модели нейронной сети для предсказания цен.
- `train.py` - скрипт для обучения модели на предобработанных данных.
- `evaluate.py` - оценка качества модели на тестовом наборе данных.

## Обучение модели

Запустите скрипт обучения модели, используя команду:

python train.py

Этот скрипт проведет обучение на обучающем наборе данных, используя архитектуру нейронной сети, определенную в `model.py`.

## Оценка модели

После обучения модели вы можете оценить ее эффективность на тестовых данных:

python evaluate.py


В результате будет выведена таблица с показателями R^2, MAE, MSE, RMSE.

