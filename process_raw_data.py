import pandas as pd
import numpy as np

#reduce the userid and itemid to only 1000 dimensions
reduced_dim = True

data = pd.read_csv('./data/df_electronics.csv', header = 0, delimiter=',')


print("Data size: ", data.shape[0])
print("Data features: ", list(data.columns))
print("Fisrt 5 rows:")
print("-" * 105)
print(data.head())
print("-" * 105)
print("Item number: ", data['item_id'].unique().shape[0])
print("User number: ", data['user_id'].unique().shape[0])
print("Category number: ", data['category'].unique().shape[0])
print("Gender number: ", data['model_attr'].unique().shape[0])


# convert gender to index
gender_to_encoding = {
    'Female' : 0,
    'Male' : 1,
    'Female&Male' : 2,
}
category_to_encoding = { name: i for i, name in enumerate(data['category'].unique())}

encoded_gender = data['model_attr'].apply(lambda x: gender_to_encoding[x])
encoded_category = data['category'].apply(lambda x: category_to_encoding[x])

if reduced_dim:
    print("You choose to reduce the dimension of users and items to 1000")
    print("Processing...")
    items = []
    for i in range(data['item_id'].shape[0]):
        items.append(data['item_id'].iloc[i] % 1000)
    items = np.array(items)

    users = []
    for i in range(data['user_id'].shape[0]):
        users.append(data['user_id'].iloc[i] % 1000)
    users = np.array(users)

    np.save("./metadata/user.npy",  users)
    np.save("./metadata/item.npy",items)
    np.save("./metadata/gender.npy", encoded_gender.to_numpy())
    np.save("./metadata/rating.npy", data['rating'].astype(int).to_numpy())
    np.save("./metadata/category.npy",encoded_category.to_numpy())
    print("Done")
else:
    np.save("./metadata/gender.npy", encoded_gender.to_numpy())
    np.save("./metadata/user.npy",  data['user_id'].astype(int).to_numpy())
    np.save("./metadata/rating.npy", data['rating'].astype(int).to_numpy())
    np.save("./metadata/item.npy",data['item_id'].astype(int).to_numpy())
    np.save("./metadata/category.npy",encoded_category.to_numpy())
