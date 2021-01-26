import pandas as pd


def prefilter_items(data_train):
    # Оставим только 5000 самых популярных товаров
    popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_5000 = popularity.sort_values(
        'n_sold', ascending=False).head(5000).item_id.tolist()
    # добавим, чтобы не потерять юзеров
    data_train.loc[~data_train['item_id'].isin(top_5000), 'item_id'] = 999999

    # Уберем самые популярные

    # Уберем самые непопулряные

    # Уберем товары, которые не продавались за последние 12 месяцев

    # Уберем не интересные для рекоммендаций категории (department)

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.

    # Уберем слишком дорогие товарыs

    # ...

    return data_train


def postfilter_items():
    pass


def test_train_creater(path_csv):
    '''
    Создает тренировочный и тестовый датасеты из csv файла
    '''
    data = pd.read_csv(path_csv)

    data.columns = [col.lower() for col in data.columns]
    data.rename(columns={'household_key': 'user_id',
                         'product_id': 'item_id'},
                inplace=True)

    test_size_weeks = 3

    data_train = data[data['week_no'] <
                      data['week_no'].max() - test_size_weeks]
    data_test = data[data['week_no'] >=
                     data['week_no'].max() - test_size_weeks]

    print(data_train.head(2))
    return data_test, data_train


def get_item_fitures(path_csv):
    '''
    Создает дополнительные признаки продукта из csv файла
    '''
    item_features = pd.read_csv(path_csv)

    item_features.columns = [col.lower() for col in item_features.columns]
    item_features.rename(columns={'product_id': 'item_id'}, inplace=True)

    print(item_features.head(2))
    print(item_features.department.unique())
    return item_features


def get_csr_matrix(data_train, user_fiture):
    '''
    Создает матрицу типа csr из data_train по признаку user_fiture
    '''
    # Делаем сводную таблицу user - строки, item - колонки, значения - сумма user_fiture
    user_item_matrix = pd.pivot_table(data_train,
                                      index='user_id', columns='item_id',
                                      values=user_fiture,  # 'quantity' Можно пробовать другие варианты
                                      aggfunc='count',
                                      fill_value=0
                                      )

    user_item_matrix = user_item_matrix.astype(
        float)  # необходимый тип матрицы для implicit

    # переведем в формат saprse matrix
    sparse_user_item = csr_matrix(user_item_matrix).tocsr()

    print(user_item_matrix.head(3))
    print(user_item_matrix.shape)
    return user_item_matrix, sparse_user_item


def get_id_matrix(user_item_matrix):
    userids = user_item_matrix.index.values
    itemids = user_item_matrix.columns.values

    matrix_userids = np.arange(len(userids))
    matrix_itemids = np.arange(len(itemids))

    id_to_itemid = dict(zip(matrix_itemids, itemids))
    id_to_userid = dict(zip(matrix_userids, userids))

    itemid_to_id = dict(zip(itemids, matrix_itemids))
    userid_to_id = dict(zip(userids, matrix_userids))
    return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
