//
// Autogenerated by Thrift Compiler (0.13.0)
//
// DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
//
"use strict";

var thrift = require('thrift');
var Thrift = thrift.Thrift;
var Q = thrift.Q;
var Int64 = require('node-int64');


var ttypes = require('./cv_types');
//HELPER FUNCTIONS AND STRUCTURES

var cvService_get_location_list_args = function(args) {
};
cvService_get_location_list_args.prototype = {};
cvService_get_location_list_args.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    input.skip(ftype);
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

cvService_get_location_list_args.prototype.write = function(output) {
  output.writeStructBegin('cvService_get_location_list_args');
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var cvService_get_location_list_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = Thrift.copyList(args.success, [null]);
    }
  }
};
cvService_get_location_list_result.prototype = {};
cvService_get_location_list_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.LIST) {
        this.success = [];
        var _rtmp320 = input.readListBegin();
        var _size19 = _rtmp320.size || 0;
        for (var _i21 = 0; _i21 < _size19; ++_i21) {
          var elem22 = null;
          elem22 = input.readString();
          this.success.push(elem22);
        }
        input.readListEnd();
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

cvService_get_location_list_result.prototype.write = function(output) {
  output.writeStructBegin('cvService_get_location_list_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.LIST, 0);
    output.writeListBegin(Thrift.Type.STRING, this.success.length);
    for (var iter23 in this.success) {
      if (this.success.hasOwnProperty(iter23)) {
        iter23 = this.success[iter23];
        output.writeString(iter23);
      }
    }
    output.writeListEnd();
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var cvService_get_snapshots_args = function(args) {
  this.location = null;
  if (args) {
    if (args.location !== undefined && args.location !== null) {
      this.location = args.location;
    }
  }
};
cvService_get_snapshots_args.prototype = {};
cvService_get_snapshots_args.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 1:
      if (ftype == Thrift.Type.STRING) {
        this.location = input.readString();
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

cvService_get_snapshots_args.prototype.write = function(output) {
  output.writeStructBegin('cvService_get_snapshots_args');
  if (this.location !== null && this.location !== undefined) {
    output.writeFieldBegin('location', Thrift.Type.STRING, 1);
    output.writeString(this.location);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var cvService_get_snapshots_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = new ttypes.SnapShots(args.success);
    }
  }
};
cvService_get_snapshots_result.prototype = {};
cvService_get_snapshots_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.STRUCT) {
        this.success = new ttypes.SnapShots();
        this.success.read(input);
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

cvService_get_snapshots_result.prototype.write = function(output) {
  output.writeStructBegin('cvService_get_snapshots_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.STRUCT, 0);
    this.success.write(output);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var cvService_get_location_info_args = function(args) {
  this.location = null;
  if (args) {
    if (args.location !== undefined && args.location !== null) {
      this.location = args.location;
    }
  }
};
cvService_get_location_info_args.prototype = {};
cvService_get_location_info_args.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 1:
      if (ftype == Thrift.Type.STRING) {
        this.location = input.readString();
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

cvService_get_location_info_args.prototype.write = function(output) {
  output.writeStructBegin('cvService_get_location_info_args');
  if (this.location !== null && this.location !== undefined) {
    output.writeFieldBegin('location', Thrift.Type.STRING, 1);
    output.writeString(this.location);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var cvService_get_location_info_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = new ttypes.LocationInfo(args.success);
    }
  }
};
cvService_get_location_info_result.prototype = {};
cvService_get_location_info_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.STRUCT) {
        this.success = new ttypes.LocationInfo();
        this.success.read(input);
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

cvService_get_location_info_result.prototype.write = function(output) {
  output.writeStructBegin('cvService_get_location_info_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.STRUCT, 0);
    this.success.write(output);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var cvServiceClient = exports.Client = function(output, pClass) {
  this.output = output;
  this.pClass = pClass;
  this._seqid = 0;
  this._reqs = {};
};
cvServiceClient.prototype = {};
cvServiceClient.prototype.seqid = function() { return this._seqid; };
cvServiceClient.prototype.new_seqid = function() { return this._seqid += 1; };

cvServiceClient.prototype.get_location_list = function(callback) {
  this._seqid = this.new_seqid();
  if (callback === undefined) {
    var _defer = Q.defer();
    this._reqs[this.seqid()] = function(error, result) {
      if (error) {
        _defer.reject(error);
      } else {
        _defer.resolve(result);
      }
    };
    this.send_get_location_list();
    return _defer.promise;
  } else {
    this._reqs[this.seqid()] = callback;
    this.send_get_location_list();
  }
};

cvServiceClient.prototype.send_get_location_list = function() {
  var output = new this.pClass(this.output);
  var args = new cvService_get_location_list_args();
  try {
    output.writeMessageBegin('get_location_list', Thrift.MessageType.CALL, this.seqid());
    args.write(output);
    output.writeMessageEnd();
    return this.output.flush();
  }
  catch (e) {
    delete this._reqs[this.seqid()];
    if (typeof output.reset === 'function') {
      output.reset();
    }
    throw e;
  }
};

cvServiceClient.prototype.recv_get_location_list = function(input,mtype,rseqid) {
  var callback = this._reqs[rseqid] || function() {};
  delete this._reqs[rseqid];
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(input);
    input.readMessageEnd();
    return callback(x);
  }
  var result = new cvService_get_location_list_result();
  result.read(input);
  input.readMessageEnd();

  if (null !== result.success) {
    return callback(null, result.success);
  }
  return callback('get_location_list failed: unknown result');
};

cvServiceClient.prototype.get_snapshots = function(location, callback) {
  this._seqid = this.new_seqid();
  if (callback === undefined) {
    var _defer = Q.defer();
    this._reqs[this.seqid()] = function(error, result) {
      if (error) {
        _defer.reject(error);
      } else {
        _defer.resolve(result);
      }
    };
    this.send_get_snapshots(location);
    return _defer.promise;
  } else {
    this._reqs[this.seqid()] = callback;
    this.send_get_snapshots(location);
  }
};

cvServiceClient.prototype.send_get_snapshots = function(location) {
  var output = new this.pClass(this.output);
  var params = {
    location: location
  };
  var args = new cvService_get_snapshots_args(params);
  try {
    output.writeMessageBegin('get_snapshots', Thrift.MessageType.CALL, this.seqid());
    args.write(output);
    output.writeMessageEnd();
    return this.output.flush();
  }
  catch (e) {
    delete this._reqs[this.seqid()];
    if (typeof output.reset === 'function') {
      output.reset();
    }
    throw e;
  }
};

cvServiceClient.prototype.recv_get_snapshots = function(input,mtype,rseqid) {
  var callback = this._reqs[rseqid] || function() {};
  delete this._reqs[rseqid];
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(input);
    input.readMessageEnd();
    return callback(x);
  }
  var result = new cvService_get_snapshots_result();
  result.read(input);
  input.readMessageEnd();

  if (null !== result.success) {
    return callback(null, result.success);
  }
  return callback('get_snapshots failed: unknown result');
};

cvServiceClient.prototype.get_location_info = function(location, callback) {
  this._seqid = this.new_seqid();
  if (callback === undefined) {
    var _defer = Q.defer();
    this._reqs[this.seqid()] = function(error, result) {
      if (error) {
        _defer.reject(error);
      } else {
        _defer.resolve(result);
      }
    };
    this.send_get_location_info(location);
    return _defer.promise;
  } else {
    this._reqs[this.seqid()] = callback;
    this.send_get_location_info(location);
  }
};

cvServiceClient.prototype.send_get_location_info = function(location) {
  var output = new this.pClass(this.output);
  var params = {
    location: location
  };
  var args = new cvService_get_location_info_args(params);
  try {
    output.writeMessageBegin('get_location_info', Thrift.MessageType.CALL, this.seqid());
    args.write(output);
    output.writeMessageEnd();
    return this.output.flush();
  }
  catch (e) {
    delete this._reqs[this.seqid()];
    if (typeof output.reset === 'function') {
      output.reset();
    }
    throw e;
  }
};

cvServiceClient.prototype.recv_get_location_info = function(input,mtype,rseqid) {
  var callback = this._reqs[rseqid] || function() {};
  delete this._reqs[rseqid];
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(input);
    input.readMessageEnd();
    return callback(x);
  }
  var result = new cvService_get_location_info_result();
  result.read(input);
  input.readMessageEnd();

  if (null !== result.success) {
    return callback(null, result.success);
  }
  return callback('get_location_info failed: unknown result');
};
var cvServiceProcessor = exports.Processor = function(handler) {
  this._handler = handler;
};
cvServiceProcessor.prototype.process = function(input, output) {
  var r = input.readMessageBegin();
  if (this['process_' + r.fname]) {
    return this['process_' + r.fname].call(this, r.rseqid, input, output);
  } else {
    input.skip(Thrift.Type.STRUCT);
    input.readMessageEnd();
    var x = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN_METHOD, 'Unknown function ' + r.fname);
    output.writeMessageBegin(r.fname, Thrift.MessageType.EXCEPTION, r.rseqid);
    x.write(output);
    output.writeMessageEnd();
    output.flush();
  }
};
cvServiceProcessor.prototype.process_get_location_list = function(seqid, input, output) {
  var args = new cvService_get_location_list_args();
  args.read(input);
  input.readMessageEnd();
  if (this._handler.get_location_list.length === 0) {
    Q.fcall(this._handler.get_location_list.bind(this._handler)
    ).then(function(result) {
      var result_obj = new cvService_get_location_list_result({success: result});
      output.writeMessageBegin("get_location_list", Thrift.MessageType.REPLY, seqid);
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    }).catch(function (err) {
      var result;
      result = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
      output.writeMessageBegin("get_location_list", Thrift.MessageType.EXCEPTION, seqid);
      result.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  } else {
    this._handler.get_location_list(function (err, result) {
      var result_obj;
      if ((err === null || typeof err === 'undefined')) {
        result_obj = new cvService_get_location_list_result((err !== null || typeof err === 'undefined') ? err : {success: result});
        output.writeMessageBegin("get_location_list", Thrift.MessageType.REPLY, seqid);
      } else {
        result_obj = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
        output.writeMessageBegin("get_location_list", Thrift.MessageType.EXCEPTION, seqid);
      }
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  }
};
cvServiceProcessor.prototype.process_get_snapshots = function(seqid, input, output) {
  var args = new cvService_get_snapshots_args();
  args.read(input);
  input.readMessageEnd();
  if (this._handler.get_snapshots.length === 1) {
    Q.fcall(this._handler.get_snapshots.bind(this._handler),
      args.location
    ).then(function(result) {
      var result_obj = new cvService_get_snapshots_result({success: result});
      output.writeMessageBegin("get_snapshots", Thrift.MessageType.REPLY, seqid);
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    }).catch(function (err) {
      var result;
      result = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
      output.writeMessageBegin("get_snapshots", Thrift.MessageType.EXCEPTION, seqid);
      result.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  } else {
    this._handler.get_snapshots(args.location, function (err, result) {
      var result_obj;
      if ((err === null || typeof err === 'undefined')) {
        result_obj = new cvService_get_snapshots_result((err !== null || typeof err === 'undefined') ? err : {success: result});
        output.writeMessageBegin("get_snapshots", Thrift.MessageType.REPLY, seqid);
      } else {
        result_obj = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
        output.writeMessageBegin("get_snapshots", Thrift.MessageType.EXCEPTION, seqid);
      }
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  }
};
cvServiceProcessor.prototype.process_get_location_info = function(seqid, input, output) {
  var args = new cvService_get_location_info_args();
  args.read(input);
  input.readMessageEnd();
  if (this._handler.get_location_info.length === 1) {
    Q.fcall(this._handler.get_location_info.bind(this._handler),
      args.location
    ).then(function(result) {
      var result_obj = new cvService_get_location_info_result({success: result});
      output.writeMessageBegin("get_location_info", Thrift.MessageType.REPLY, seqid);
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    }).catch(function (err) {
      var result;
      result = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
      output.writeMessageBegin("get_location_info", Thrift.MessageType.EXCEPTION, seqid);
      result.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  } else {
    this._handler.get_location_info(args.location, function (err, result) {
      var result_obj;
      if ((err === null || typeof err === 'undefined')) {
        result_obj = new cvService_get_location_info_result((err !== null || typeof err === 'undefined') ? err : {success: result});
        output.writeMessageBegin("get_location_info", Thrift.MessageType.REPLY, seqid);
      } else {
        result_obj = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
        output.writeMessageBegin("get_location_info", Thrift.MessageType.EXCEPTION, seqid);
      }
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  }
};
