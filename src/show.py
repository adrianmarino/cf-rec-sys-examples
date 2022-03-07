def show_similar_items(indices, similarities, item_id):
    print(f'{len(similarities)} most similar items to item {item_id}:\n')
    for curr_item_id, sim in zip(indices+1, similarities):
        print(f'- Item ID {curr_item_id}, with similarity of {sim}.')

def show_similar_users(indices, similarities, user_id):
    print(f'{len(similarities)} most similar users to user {user_id}:\n')
    for curr_user_id, sim in zip(indices+1, similarities):
        print(f'- User ID {curr_user_id}, with similarity of {sim}.')

def show_prediction(rm, user_id, item_id, prediction):
    rating  = rm.cell(user_id, item_id)
    print(f'Item base rating prediction:')
    print(f'- User ID: {user_id}.')
    print(f'- Item ID: {item_id}.')
    print(f'- Predicted rating: {prediction}.')
    print(f'- Real rating: {rating} (0 == unrated item).')