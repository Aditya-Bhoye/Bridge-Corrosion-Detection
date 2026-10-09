import test from 'node:test';
import assert from 'node:assert/strict';
import {validateFile,severity,validateResult,MAX_SIZE} from '../src/detection.js';
test('rejects unsupported, empty and oversized uploads',()=>{assert.ok(validateFile({type:'text/plain',size:10}));assert.ok(validateFile({type:'image/png',size:0}));assert.ok(validateFile({type:'image/jpeg',size:MAX_SIZE+1}));assert.equal(validateFile({type:'image/webp',size:100}),'');});
test('coverage bands include boundaries',()=>{assert.equal(severity(9.99),'Low');assert.equal(severity(10),'Moderate');assert.equal(severity(25),'High');});
test('invalid measurements fail rather than enter history',()=>{for(const coverage of [NaN,-1,101,'20']) assert.throws(()=>validateResult({coverage}));assert.equal(validateResult({coverage:20}).severity,'Moderate');});
