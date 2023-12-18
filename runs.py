import copy
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem


# Basic Paths:
path_inputs_file = './input_file.txt'
path_outputs_dir = './output'
path_drugs_ids_file = './dataset/columns_ids.csv'
path_sdf_dir = './dataset/drugbank_v5_10_sdf'

path_model_file = './model/deepddidfi.h5'
path_mlb_file = './model/mlb_111.pkl'
path_pca_file = './model/pca_86.pkl'
path_scalar_file = './model/scalar.pkl'
path_sentences_file = './dataset/ddi_sentences_111.csv'


# preprocessing:
def get_mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol

def get_mol_from_sdf(sdf_path):
    mol = Chem.MolFromMolFile(sdf_path)
    return mol

def prepare_input_data(path_input, path_compare_target_drugs_file, path_output_file, path_scalar, path_pca, path_sdf):
    df_input = pd.read_csv(path_input)  # columns = ['name1', 'smiles1', 'name2', 'smiles2']
    df_drugs = pd.read_csv(path_compare_target_drugs_file)  # columns = ['id', 'name']
    uniqe_drugs = {}
    for i in df_input.index:
        name1, name2 = df_input['name1'].iloc[i], df_input['name2'].iloc[i]
        smi1, smi2 = df_input['smiles1'].iloc[i], df_input['smiles2'].iloc[i]
        if name1 not in uniqe_drugs:
            uniqe_drugs[name1] = smi1
        if name2 not in uniqe_drugs:
            uniqe_drugs[name2] = smi2
    ssp = {}
    for i in uniqe_drugs:
        ssp[i] = {}
        ssp[i]['Error'] = 0
        smi = uniqe_drugs[i]
        try:
            mol1 = get_mol_from_smiles(smi)
            mol1 = AllChem.AddHs(mol1)
            fps1 = AllChem.GetMorganFingerprint(mol1, 4)
            for j in df_drugs['id'].values.tolist():
                mol2 = get_mol_from_sdf(f'{path_sdf}/{j}.sdf')
                mol2 = AllChem.AddHs(mol2)
                fps2 = AllChem.GetMorganFingerprint(mol2, 4)
                score = DataStructs.TanimotoSimilarity(fps1, fps2)
                ssp[i][j] = score
        except ValueError as e:
            ssp[i]['Error'] = 1
            print(f"Error: {e}")
    dfssp = pd.DataFrame.from_dict(ssp).T
    valid_pairs = []
    for i in df_input.index:
        name1, name2 = df_input['name1'].iloc[i], df_input['name2'].iloc[i]
        er1, er2 = dfssp['Error'].loc[name1], dfssp['Error'].loc[name2]
        if er1 == 1 or er2 == 1:
            continue
        else:
            valid_pairs.append((name1, name2))
    dfpairs = pd.DataFrame(valid_pairs, columns=['name1', 'name2'])
    dfssp = dfssp[dfssp['Error']==0].drop(columns=['Error'])
    with open(path_scalar, 'rb') as f:
        scalar = pickle.load(f)
    dfssp_scaled = pd.DataFrame(scalar.transform(dfssp.values))
    with open(path_pca, 'rb') as f:
        pca = pickle.load(f)
    pca_values = pca.transform(dfssp_scaled.values)
    dfpca = pd.DataFrame(pca_values)
    dfpca.index = dfssp.index
    dfpca.columns = [f'PC_{i}' for i in range(1, 87)]
    
    final_ssp = {}
    for i in dfpairs.index:
        d1, d2 = dfpairs['name1'].iloc[i], dfpairs['name2'].iloc[i]
        key = f'{d1}_{d2}'
        final_ssp[key] = []
        for j in dfpca.loc[d1].values:
            final_ssp[key].append(j)
        for j in dfpca.loc[d2].values:
            final_ssp[key].append(j)
    dffinal = pd.DataFrame(final_ssp).T
    dffinal.columns = [f'PC_{i}' for i in range(1, 173)]
    dffinal.to_csv(path_output_file)


# prediction:
def predect(path_model, path_final_pca, path_output_file, path_mlb, threshold):
    model = tf.keras.models.load_model(path_model)
    with open(path_mlb, 'rb') as f:
        mlb = pickle.load(f)
    dfdata = pd.read_csv(path_final_pca)
    dataset = dfdata.values[:, 1:].astype('float32')
    dataset = tf.convert_to_tensor(dataset, dtype=tf.float32)
    
    iter_num = 2
    predictions = []
    for i in range(iter_num):
        y_predicted = model.predict(dataset)
        predictions.append(y_predicted)
    arr = np.asarray(predictions)
    y_predicted_mean = np.mean(arr, axis=0)
    original_predicted_ddi = copy.deepcopy(y_predicted_mean)
    y_predicted_mean[y_predicted_mean >= threshold] = 1
    y_predicted_mean[y_predicted_mean < threshold] = 0
    y_predicted_inverse = mlb.inverse_transform(y_predicted_mean)
    
    fp = open(path_output_file, 'w')
    fp.write('drug pair,predicted ddi_type,score\n')
    for i in range(dfdata.shape[0]):
        predicted_ddi_score = original_predicted_ddi[i]
        predicted_ddi = y_predicted_inverse[i]
        each_ddi = dfdata.iloc[i, 0]
        for each_predicted_ddi in predicted_ddi:
            fp.write(f'{each_ddi},{each_predicted_ddi},{predicted_ddi_score[each_predicted_ddi-1]}\n')
    fp.close()


# result processing:
def result_processing(path_sentences, path_result, path_output_file):
    dfres = pd.read_csv(path_result)
    dfsent = pd.read_csv(path_sentences)
    with open(path_output_file, 'w') as f:
        f.write('drug1,drug2,interaction,score\n')
        for i in dfres.index:
            ddi_type = dfres['predicted ddi_type'].iloc[i]
            drugs = dfres['drug pair'].iloc[i]
            score = dfres['score'].iloc[i]
            drug1, drug2 = drugs.split('_')[0], drugs.split('_')[1]
            sentence = dfsent[dfsent['ddi_type'] == ddi_type]['sentences'].values[0]
            sentence = sentence.replace('#drug1', drug1).replace('#drug2', drug2)
            f.write(f'{drug1},{drug2},{sentence},{score}\n')





# Run:
path_output_pca_file = f'{path_outputs_dir}/pca_86_file.csv'
prepare_input_data(path_inputs_file, path_drugs_ids_file, path_output_pca_file, path_scalar_file, path_pca_file, path_sdf_dir)

path_output_result_file = f'{path_outputs_dir}/result_v01.csv'
predect(path_model_file, path_output_pca_file, path_output_result_file, path_mlb_file, 0.4)

path_output_final_result_file = f'{path_outputs_dir}/result_v02.csv'
result_processing(path_sentences_file, path_output_result_file, path_output_final_result_file)
