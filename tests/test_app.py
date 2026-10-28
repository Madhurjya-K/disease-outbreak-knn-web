import io
import os
from pathlib import Path
import tempfile
from fastapi.testclient import TestClient
from app.main import app, STATE, validate_csv, DATA
import pytest

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c: yield c

def vector():
    from app.model import experiment
    v={s:0 for s in experiment.SYMPTOMS}
    v.update(fever=1,cough=1,fatigue=1,breathlessness=1,loss_of_smell=1,age=58,temperature_c=38.9,days_since_onset=4,contact_with_case=1)
    return v

@pytest.mark.parametrize('path',['/','/assess','/experiments','/dataset','/methodology'])
def test_pages(client,path):
    r=client.get(path)
    assert r.status_code==200 and 'outbreak' in r.text
    assert r.headers['x-frame-options']=='DENY'
    assert "script-src 'self'" in r.headers['content-security-policy']

def test_health_and_model(client):
    assert client.get('/api/health').json()['model_ready']
    m=client.get('/api/model').json()
    assert m['samples']==3000 and len(m['features'])==18
    assert m['train_size']==2400 and m['test_size']==600
    assert 0<m['accuracy']<1 and 0<m['macro_f1']<1
    assert sum(sum(row) for row in m['confusion_matrix'])==600

def test_prediction(client):
    before=set(DATA.rglob('*'))
    r=client.post('/api/predict',json=vector())
    assert r.status_code==200
    j=r.json()
    assert j['label']=='High'
    assert abs(sum(j['vote_shares'].values())-1)<1e-10
    assert set(DATA.rglob('*'))==before
    assert client.post('/api/predict',json=vector()).json()==j

@pytest.mark.parametrize('field,value',[('age',0),('age',121),('temperature_c',50),('days_since_onset',-1),('fever',2),('fever',.5),('cough',True),('days_since_onset',1.5),('contact_with_case',-1)])
def test_invalid_vectors(client,field,value):
    v=vector();v[field]=value
    assert client.post('/api/predict',json=v).status_code==422

def test_missing_extra(client):
    v=vector();v.pop('fever')
    assert client.post('/api/predict',json=v).status_code==422
    v=vector();v['patient_name']='Must not store'
    assert client.post('/api/predict',json=v).status_code==422

def test_dataset(client):
    r=client.get('/api/dataset')
    assert r.status_code==200 and 'attachment' in r.headers['content-disposition']
    assert len(r.text.splitlines())==3001
    j=client.post('/api/dataset/validate',json={'csv_text':r.text}).json()
    assert j['valid'] and j['rows']==3000

def test_text_labels(client):
    df=STATE['dataset'].copy();df['risk']=df.risk.map({0:'Low',1:'Medium',2:'High'})
    assert len(validate_csv(df.to_csv(index=False)))==3000

@pytest.mark.parametrize('kind',['missing','bad_flag','nan','one_class','too_small','fractional_day','infinity','extra'])
def test_csv_rejection(client,kind):
    df=STATE['dataset'].copy()
    if kind=='missing':df=df.drop(columns='fever')
    if kind=='extra':df['name']='private'
    if kind=='bad_flag':df.loc[0,'fever']=3
    if kind=='nan':df.loc[0,'age']=float('nan')
    if kind=='infinity':df.loc[0,'age']=float('inf')
    if kind=='one_class':df['risk']=0
    if kind=='too_small':df=df.head(20)
    if kind=='fractional_day':df['days_since_onset']=1.5
    assert client.post('/api/dataset/validate',json={'csv_text':df.to_csv(index=False)}).status_code==422

def test_artifact_allowlist(client):
    assert client.get('/api/runs/not-an-id').status_code==404
    assert client.get('/api/runs/'+'0'*32+'/plots/secrets.txt').status_code==404
    assert client.get('/api/runs/'+'0'*32+'/results').status_code==404

def test_cross_origin(client):
    assert client.post('/api/runs',json={},headers={'Origin':'https://evil.example'}).status_code==403

def test_remote_rejected():
    with TestClient(app,client=('192.0.2.10',123)) as c:
        assert c.post('/api/runs',json={}).status_code==403

def test_operator_key(client,monkeypatch):
    monkeypatch.setenv('APP_API_TOKEN','test-only-key')
    assert client.post('/api/dataset/validate',json={}).status_code==403
    assert client.post('/api/dataset/validate',json={},headers={'X-API-Key':'test-only-key'}).status_code==422

def test_body_limit(client):
    assert client.post('/api/predict',content='x'*2_000_001).status_code==413

def test_no_fake_assets(client):
    assert client.get('/static/app.js').status_code==200
    assert client.get('/static/style.css').status_code==200
    assert client.get('/unknown').status_code==404
