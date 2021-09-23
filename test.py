
pred_l = 0.8
pred_r = 0.1

state_l = f'O [{pred_l}%]' if pred_l > 0.1 else f'- [{pred_l}%]'
state_r = f'O [{pred_r}%]' if pred_r > 0.1 else f'- [{pred_r}%]'

print(f"steate_l : {state_l}, state_r : {state_r}")