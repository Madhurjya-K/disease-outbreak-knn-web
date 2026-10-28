const fs=require('node:fs');
const assert=require('node:assert/strict');
const {JSDOM}=require('jsdom');
const base=process.env.TEST_BASE_URL||'http://127.0.0.1:8765';
const script=fs.readFileSync('app/static/app.js','utf8');
const html=fs.readFileSync('app/templates/index.html','utf8');
const checks=[];
function ok(value,message){assert(value,message);checks.push(message)}
async function until(test){const start=Date.now();while(!test()){if(Date.now()-start>20000)throw Error('Timed out waiting for UI state');await new Promise(r=>setTimeout(r,50))}}
async function page(path){const d=new JSDOM(html,{url:base+path,runScripts:'outside-only',pretendToBeVisual:true});d.window.fetch=(p,o)=>fetch(new URL(p,base),o);await d.window.eval('(async()=>{'+script+'})()');return d}
(async()=>{
let d=await page('/');
ok(d.window.document.querySelectorAll('nav a').length===5,'Five navigation routes');
ok(d.window.document.querySelectorAll('#stats .card').length===3,'Overview renders actual model statistics');
ok(!d.window.document.querySelector('.error'),'Overview has no displayed API error');d.window.close();
d=await page('/assess');let doc=d.window.document;
ok(doc.querySelectorAll('#symptoms input').length===14,'All fourteen symptom inputs render');
doc.querySelector('#demo').click();ok(doc.querySelector('[name=age]').value==='58','Report preset fills assessment form');
doc.querySelector('form').dispatchEvent(new d.window.Event('submit',{bubbles:true,cancelable:true}));
await until(()=>doc.querySelector('#result .result-label'));
ok(doc.querySelector('.result-label').textContent==='High','Real report example predicts High');
ok(doc.querySelectorAll('#result progress').length===3,'All three vote shares render');
doc.querySelector('[type=reset]').click();ok(!doc.querySelector('.result-label'),'Reset clears previous result');
doc.querySelector('[name=age]').value='0';doc.querySelector('form').dispatchEvent(new d.window.Event('submit',{bubbles:true,cancelable:true}));
await until(()=>doc.querySelector('#result .error'));ok(!!doc.querySelector('#result .error'),'Server validation errors render in assessment');d.window.close();
d=await page('/dataset');doc=d.window.document;
ok(doc.querySelector('#schema').textContent.includes('temperature_c'),'CSV schema shown');
const csv=await(await fetch(base+'/api/dataset')).text();
const lines=csv.trim().split(/\r?\n/);const small=[lines[0],...lines.slice(1,201)].join('\n');
const file={size:Buffer.byteLength(small),text:async()=>small};
Object.defineProperty(doc.querySelector('#file'),'files',{value:[file]});
doc.querySelector('#consent').checked=true;
doc.querySelector('form').dispatchEvent(new d.window.Event('submit',{bubbles:true,cancelable:true}));
await until(()=>!doc.querySelector('#train-csv').disabled);
ok(doc.querySelector('#csv-result').textContent.includes('200 rows validated'),'CSV form validates real uploaded data');
// Native file validity differs in jsdom; native browser validity remains in production code.
doc.querySelector('form').reportValidity=()=>true;
doc.querySelector('#train-csv').click();await until(()=>doc.querySelector('#csv-result a'));
ok(doc.querySelector('#csv-result a').getAttribute('href')==='/experiments','CSV form queues actual backend job and links to history');d.window.close();
d=await page('/experiments');doc=d.window.document;
ok(doc.querySelectorAll('#runs .run').length>=1,'Experiment history renders persistent backend records');d.window.close();
d=await page('/methodology');ok(d.window.document.querySelector('main').textContent.includes('not calibrated'),'Methodology includes model-calibration limitation');d.window.close();
console.log(JSON.stringify({passed:checks.length,checks},null,2));
})().catch(e=>{console.error(e);process.exit(1)});
