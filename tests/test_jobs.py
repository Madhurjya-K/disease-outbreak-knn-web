from fastapi.testclient import TestClient
from app.main import app, POOL, db

def test_queue_and_result_states(monkeypatch):
    # Execute no real worker here; live integration tests execute actual experiments.
    monkeypatch.setattr(POOL,'submit',lambda *args,**kwargs:None)
    with TestClient(app) as c:
        with db() as conn: conn.execute('DELETE FROM runs')
        a=c.post('/api/runs',json={})
        b=c.post('/api/runs',json={})
        assert a.status_code==b.status_code==202
        id=a.json()['id']
        assert c.get('/api/runs/'+id).json()['status']=='queued'
        assert c.get('/api/runs/'+id+'/results').status_code==409
        assert c.get('/api/runs/'+id+'/plots/01_eda.png').status_code==409
        assert c.post('/api/runs',json={}).status_code==429
        assert len(c.get('/api/runs').json())==2
    with TestClient(app) as c:
        rows=c.get('/api/runs').json()
        assert len(rows)==2 and all(r['status']=='failed' for r in rows)
        assert all('restarted' in r['error'] for r in rows)
        with db() as conn: conn.execute('DELETE FROM runs')

def test_invalid_upload_not_queued():
    with TestClient(app) as c:
        before=c.get('/api/runs').json()
        assert c.post('/api/runs',json={'csv_text':'invalid'}).status_code==422
        assert c.get('/api/runs').json()==before
