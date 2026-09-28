import json
import pickle
import os
import pandas as pd
from tqdm import tqdm
import torch
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset(dataset):
	dataset_alias = {'psm': 'PSM', 'exathlon': 'Exathlon'}
	dataset_name = dataset_alias.get(dataset.lower(), dataset)
	folder = os.path.join(output_folder, dataset_name)
	if not os.path.exists(folder):
		raise Exception(f"Processed Data not found for {dataset}.")
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset_name == 'SMD': file = 'machine-1-1_' + file
		if dataset_name == 'SMAP': file = 'P-1_' + file
		if dataset_name == 'MSL': file = 'C-1_' + file
		if dataset_name == 'UCR': file = '136_' + file
		if dataset_name == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
	if args.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = (model_class(dims).float().to(DEVICE) if "TranAD" in modelname else model_class(dims).double().to(DEVICE))
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname, map_location=DEVICE)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def save_final_result(result):
	folder = f'results/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = os.path.join(folder, 'final_result.json')
	serializable_result = {
		key: value.item() if isinstance(value, np.generic) else value
		for key, value in result.items()
	}
	with open(file_path, 'w', encoding='utf-8') as f:
		json.dump(serializable_result, f, indent=2, ensure_ascii=False)
	return file_path

def create_step_progress(epoch, data, model, training):
	if not training or model.name not in ['DAGMM', 'LSTM_AD', 'OmniAnomaly', 'MAD_GAN', 'TranAD', 'GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		return enumerate(data), None
	desc = f'Epoch {epoch} steps'
	step_bar = tqdm(enumerate(data), total=len(data), desc=desc, leave=False, mininterval=5.0)
	return step_bar, step_bar

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	device = next(model.parameters()).device
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1]
	if 'DAGMM' in model.name:
		l = nn.MSELoss(reduction = 'none')
		compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
		n = epoch + 1; w_size = model.n_window
		l1s = []; l2s = []
		if training:
			for d in data:
				_, x_hat, z, gamma = model(d)
				l1, l2 = l(x_hat, d), l(gamma, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1) + torch.mean(l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s = []
			for d in data: 
				_, x_hat, _, _ = model(d)
				ae1s.append(x_hat)
			ae1s = torch.stack(ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	if 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []; res = []
		if training:
			for d in data:
				ae, ats = model(d)
				# res.append(torch.mean(ats, axis=0).view(-1))
				l1 = l(ae, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# res = torch.stack(res); np.save('ascores.npy', res.detach().cpu().numpy())
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			ae1s, y_pred = [], []
			for d in data: 
				ae1 = model(d)
				y_pred.append(ae1[-1])
				ae1s.append(ae1)
			ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
			loss = torch.mean(l(ae1s, data), axis=1)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	elif 'OmniAnomaly' in model.name:
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				hidden = hidden.detach()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d in data:
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	# elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
	# 	l = nn.MSELoss(reduction = 'none')
	# 	n = epoch + 1; w_size = model.n_window
	# 	l1s = []
	# 	if training:
	# 		h = None
	# 		step_iter, step_bar = create_step_progress(epoch, data, model, training)
	# 		for i, d in step_iter:
	# 			if 'MTAD_GAT' in model.name: 
	# 				x, h = model(d, h)
	# 			else:
	# 				x = model(d)
	# 			loss = torch.mean(l(x, d))
	# 			l1s.append(torch.mean(loss).item())
	# 			optimizer.zero_grad()
	# 			loss.backward()
	# 			optimizer.step()
	# 			if 'MTAD_GAT' in model.name and h is not None:
	# 				h = h.detach()
	# 			if step_bar is not None and ((i + 1) % 50 == 0 or (i + 1) == len(data)):
	# 				step_bar.set_postfix(avg_mse=f'{np.mean(l1s):.6f}')
	# 		if step_bar is not None:
	# 			step_bar.close()
	# 		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
	# 		return np.mean(l1s), optimizer.param_groups[0]['lr']
	# 	else:
	# 		batch_size = model.batch if hasattr(model, 'batch') else 256
	# 		loss_chunks, pred_chunks = [], []
	# 		with torch.no_grad():
	# 			for start in range(0, len(data), batch_size):
	# 				data_batch = data[start:start + batch_size]
	# 				xs = []
	# 				if 'MTAD_GAT' in model.name:
	# 					data_batch = data_batch.to(device)
	# 				for d in data_batch:
	# 					if 'MTAD_GAT' in model.name:
	# 						x, h = model(d, None)
	# 					else:
	# 						x = model(d)
	# 					xs.append(x.detach().cpu())
	# 				xs = torch.stack(xs)
	# 				data_cpu = data_batch.detach().cpu()
	# 				pred_batch = xs[:, data_cpu.shape[1]-feats:data_cpu.shape[1]].view(-1, feats)
	# 				loss_batch = l(xs, data_cpu)
	# 				loss_batch = loss_batch[:, data_cpu.shape[1]-feats:data_cpu.shape[1]].view(-1, feats)
	# 				pred_chunks.append(pred_batch)
	# 				loss_chunks.append(loss_batch)
	# 		loss = torch.cat(loss_chunks, dim=0).numpy()
	# 		y_pred = torch.cat(pred_chunks, dim=0).numpy()
	# 		return loss, y_pred
	elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []
		if training:
			for i, d in enumerate(data):
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, h if i else None)
				else:
					x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				if 'MTAD_GAT' in model.name and h is not None:
					h = h.detach()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			if 'MTAD_GAT' in model.name:
				batch_size = model.batch if hasattr(model, 'batch') else 256
				loss_chunks, pred_chunks = [], []
				with torch.no_grad():
					for start in range(0, len(data), batch_size):
						data_batch = data[start:start + batch_size].to(device)
						xs = []
						for d in data_batch:
							x, h = model(d, None)
							xs.append(x)
						xs = torch.stack(xs)
						pred_batch = xs[:, data_batch.shape[1]-feats:data_batch.shape[1]].view(-1, feats)
						loss_batch = l(xs, data_batch)
						loss_batch = loss_batch[:, data_batch.shape[1]-feats:data_batch.shape[1]].view(-1, feats)
						pred_chunks.append(pred_batch.detach().cpu())
						loss_chunks.append(loss_batch.detach().cpu())
				loss = torch.cat(loss_chunks, dim=0).numpy()
				y_pred = torch.cat(pred_chunks, dim=0).numpy()
				return loss, y_pred
			xs = []
			for d in data: 
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, None)
				else:
					x = model(d)
				xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	elif 'GAN' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bcel = nn.BCELoss(reduction = 'mean')
		msel = nn.MSELoss(reduction = 'mean')
		real_label, fake_label = torch.tensor([0.9], dtype=torch.double, device=device), torch.tensor([0.1], dtype=torch.double, device=device) # label smoothing
		n = epoch + 1; w_size = model.n_window
		mses, gls, dls = [], [], []
		if training:
			for d in data:
				# training discriminator
				model.discriminator.zero_grad()
				_, real, fake = model(d)
				dl = bcel(real, real_label) + bcel(fake, fake_label)
				dl.backward()
				model.generator.zero_grad()
				optimizer.step()
				# training generator
				z, _, fake = model(d)
				mse = msel(z, d) 
				gl = bcel(fake, real_label)
				tl = gl + mse
				tl.backward()
				model.discriminator.zero_grad()
				optimizer.step()
				mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
				# tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
			return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
		else:
			outputs = []
			for d in data: 
				z, _, _ = model(d)
				outputs.append(z)
			outputs = torch.stack(outputs)
			y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(outputs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = data.float().to(device) if torch.is_tensor(data) else torch.tensor(data, dtype=torch.float32, device=device)
		dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			for d, _ in dataloader:
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
			return loss.detach().cpu().numpy(), z.detach().cpu().numpy()[0]
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
			
if __name__ == '__main__':
	print(f'Using device: {DEVICE}')
	train_loader, test_loader, labels = load_dataset(args.dataset)
	print("labels shape:", labels.shape)
	print("labels unique:", np.unique(labels, return_counts=True))
	print("labelsFinal unique:", np.unique((np.sum(labels, axis=1) >= 1).astype(int), return_counts=True))
	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
	if model.name == 'MTAD_GAT':
		trainD, testD = trainD.to(DEVICE), testD
	else:
		trainD, testD = trainD.to(DEVICE), testD.to(DEVICE)
	trainO, testO = trainO.to(DEVICE), testO.to(DEVICE)

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 5; e = epoch + 1; start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	with torch.no_grad():
		loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
	print("loss shape:", loss.shape)
	print("labels shape:", labels.shape)

	### Plot curves
	if not args.test:
		if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
		plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

	### Scores
	# results = []
	with torch.no_grad():
		lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	# for i in range(loss.shape[1]):  逐维度
	# 	lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
	# 	result, pred = pot_eval(lt, l, ls); preds.append(pred)
	# 	results.append(result)
	# df = pd.DataFrame(results)
	# preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
	# pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
	lossTfinal = np.mean(lossT, axis=1)
	lossFinal = np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	print("normal loss mean:", lossFinal[labelsFinal == 0].mean())
	print("anom loss mean:", lossFinal[labelsFinal == 1].mean())
	print("normal quantile:", np.percentile(lossFinal[labelsFinal == 0], [50, 90, 95, 99]))
	print("anom quantile:", np.percentile(lossFinal[labelsFinal == 1], [50, 90, 95, 99]))
	
	for shift in [-100, -50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50, 100]:
		shifted = np.roll(labelsFinal, shift)
		print(
			"shift", shift,
			"normal:", lossFinal[shifted == 0].mean(),
			"anom:", lossFinal[shifted == 1].mean()
		)
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	result.update(hit_att(loss, labels))
	result.update(ndcg(loss, labels))
	result_path = save_final_result(result)
	# print(df)
	pprint(result)
	print(f'Saved final result to {result_path}')
	# pprint(getresults2(df, result))
	# beep(4)
