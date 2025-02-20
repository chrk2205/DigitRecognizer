# def evaluate_model_one_epoch(model, criterion, x_val, y_val):
#     model.eval()
#     val_loss = 0
#     hits = 0
#     misses = 0
#     for i in range(len(x_val)):
#         y = model(x_val[i])
#         loss = criterion(y, y_val[i])
#         val_loss += loss.item()
#         if torch.argmax(y) == torch.argmax(y_val[i]):
#             hits += 1
#         else:
#             misses += 1

#     val_loss /= len(x_val)
#     accuracy = hits / (hits + misses)
#     return val_loss, accuracy



# def train_model_one_epoch(model, optimizer, criterion, x_train, y_train):
#     """
#     Saturated at
#         tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], grad_fn=<SoftmaxBackward0>)
#     """

#     model.train()
#     train_loss = 0
#     for i in range(len(x_train)):
#         x = x_train[i]
#         y = y_train[i]
#         optimizer.zero_grad()
#         output = model(x)
#         loss = criterion(output, y)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     train_loss /= len(x_train)
#     return train_loss