// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow/tools/tfprof/tfprof_log.proto
//
// For information on using the generated types, please see the documenation:
//   https://github.com/apple/swift-protobuf/

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that your are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

/// It specifies the Python callstack that creates an op.
public struct Tensorflow_Tfprof_CodeDef: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".CodeDef"

  public var traces: [Tensorflow_Tfprof_CodeDef.Trace] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public struct Trace: SwiftProtobuf.Message {
    public static let protoMessageName: String = Tensorflow_Tfprof_CodeDef.protoMessageName + ".Trace"

    public var file: String {
      get {return _file ?? String()}
      set {_file = newValue}
    }
    /// Returns true if `file` has been explicitly set.
    public var hasFile: Bool {return self._file != nil}
    /// Clears the value of `file`. Subsequent reads from it will return its default value.
    public mutating func clearFile() {self._file = nil}

    public var lineno: Int32 {
      get {return _lineno ?? 0}
      set {_lineno = newValue}
    }
    /// Returns true if `lineno` has been explicitly set.
    public var hasLineno: Bool {return self._lineno != nil}
    /// Clears the value of `lineno`. Subsequent reads from it will return its default value.
    public mutating func clearLineno() {self._lineno = nil}

    public var function: String {
      get {return _function ?? String()}
      set {_function = newValue}
    }
    /// Returns true if `function` has been explicitly set.
    public var hasFunction: Bool {return self._function != nil}
    /// Clears the value of `function`. Subsequent reads from it will return its default value.
    public mutating func clearFunction() {self._function = nil}

    public var line: String {
      get {return _line ?? String()}
      set {_line = newValue}
    }
    /// Returns true if `line` has been explicitly set.
    public var hasLine: Bool {return self._line != nil}
    /// Clears the value of `line`. Subsequent reads from it will return its default value.
    public mutating func clearLine() {self._line = nil}

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public init() {}

    /// Used by the decoding initializers in the SwiftProtobuf library, not generally
    /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
    /// initializers are defined in the SwiftProtobuf library. See the Message and
    /// Message+*Additions` files.
    public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
      while let fieldNumber = try decoder.nextFieldNumber() {
        switch fieldNumber {
        case 1: try decoder.decodeSingularStringField(value: &self._file)
        case 2: try decoder.decodeSingularInt32Field(value: &self._lineno)
        case 3: try decoder.decodeSingularStringField(value: &self._function)
        case 4: try decoder.decodeSingularStringField(value: &self._line)
        default: break
        }
      }
    }

    /// Used by the encoding methods of the SwiftProtobuf library, not generally
    /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
    /// other serializer methods are defined in the SwiftProtobuf library. See the
    /// `Message` and `Message+*Additions` files.
    public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
      if let v = self._file {
        try visitor.visitSingularStringField(value: v, fieldNumber: 1)
      }
      if let v = self._lineno {
        try visitor.visitSingularInt32Field(value: v, fieldNumber: 2)
      }
      if let v = self._function {
        try visitor.visitSingularStringField(value: v, fieldNumber: 3)
      }
      if let v = self._line {
        try visitor.visitSingularStringField(value: v, fieldNumber: 4)
      }
      try unknownFields.traverse(visitor: &visitor)
    }

    fileprivate var _file: String? = nil
    fileprivate var _lineno: Int32? = nil
    fileprivate var _function: String? = nil
    fileprivate var _line: String? = nil
  }

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedMessageField(value: &self.traces)
      default: break
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.traces.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.traces, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }
}

public struct Tensorflow_Tfprof_OpLogEntry: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".OpLogEntry"

  /// op name.
  public var name: String {
    get {return _storage._name ?? String()}
    set {_uniqueStorage()._name = newValue}
  }
  /// Returns true if `name` has been explicitly set.
  public var hasName: Bool {return _storage._name != nil}
  /// Clears the value of `name`. Subsequent reads from it will return its default value.
  public mutating func clearName() {_storage._name = nil}

  /// float_ops is filled by tfprof Python API when called. It requires the
  /// op has RegisterStatistics defined. Currently, Conv2D, MatMul, etc, are
  /// implemented.
  public var floatOps: Int64 {
    get {return _storage._floatOps ?? 0}
    set {_uniqueStorage()._floatOps = newValue}
  }
  /// Returns true if `floatOps` has been explicitly set.
  public var hasFloatOps: Bool {return _storage._floatOps != nil}
  /// Clears the value of `floatOps`. Subsequent reads from it will return its default value.
  public mutating func clearFloatOps() {_storage._floatOps = nil}

  /// User can define extra op type information for an op. This allows the user
  /// to select a group of ops precisely using op_type as a key.
  public var types: [String] {
    get {return _storage._types}
    set {_uniqueStorage()._types = newValue}
  }

  /// Used to support tfprof "code" view.
  public var codeDef: Tensorflow_Tfprof_CodeDef {
    get {return _storage._codeDef ?? Tensorflow_Tfprof_CodeDef()}
    set {_uniqueStorage()._codeDef = newValue}
  }
  /// Returns true if `codeDef` has been explicitly set.
  public var hasCodeDef: Bool {return _storage._codeDef != nil}
  /// Clears the value of `codeDef`. Subsequent reads from it will return its default value.
  public mutating func clearCodeDef() {_storage._codeDef = nil}

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    _ = _uniqueStorage()
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      while let fieldNumber = try decoder.nextFieldNumber() {
        switch fieldNumber {
        case 1: try decoder.decodeSingularStringField(value: &_storage._name)
        case 2: try decoder.decodeSingularInt64Field(value: &_storage._floatOps)
        case 3: try decoder.decodeRepeatedStringField(value: &_storage._types)
        case 4: try decoder.decodeSingularMessageField(value: &_storage._codeDef)
        default: break
        }
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      if let v = _storage._name {
        try visitor.visitSingularStringField(value: v, fieldNumber: 1)
      }
      if let v = _storage._floatOps {
        try visitor.visitSingularInt64Field(value: v, fieldNumber: 2)
      }
      if !_storage._types.isEmpty {
        try visitor.visitRepeatedStringField(value: _storage._types, fieldNumber: 3)
      }
      if let v = _storage._codeDef {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 4)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  fileprivate var _storage = _StorageClass.defaultInstance
}

public struct Tensorflow_Tfprof_OpLog: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".OpLog"

  public var logEntries: [Tensorflow_Tfprof_OpLogEntry] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedMessageField(value: &self.logEntries)
      default: break
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.logEntries.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.logEntries, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow.tfprof"

extension Tensorflow_Tfprof_CodeDef: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "traces"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Tfprof_CodeDef) -> Bool {
    if self.traces != other.traces {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Tfprof_CodeDef.Trace: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "file"),
    2: .same(proto: "lineno"),
    3: .same(proto: "function"),
    4: .same(proto: "line"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Tfprof_CodeDef.Trace) -> Bool {
    if self._file != other._file {return false}
    if self._lineno != other._lineno {return false}
    if self._function != other._function {return false}
    if self._line != other._line {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Tfprof_OpLogEntry: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "name"),
    2: .standard(proto: "float_ops"),
    3: .same(proto: "types"),
    4: .standard(proto: "code_def"),
  ]

  fileprivate class _StorageClass {
    var _name: String? = nil
    var _floatOps: Int64? = nil
    var _types: [String] = []
    var _codeDef: Tensorflow_Tfprof_CodeDef? = nil

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _name = source._name
      _floatOps = source._floatOps
      _types = source._types
      _codeDef = source._codeDef
    }
  }

  fileprivate mutating func _uniqueStorage() -> _StorageClass {
    if !isKnownUniquelyReferenced(&_storage) {
      _storage = _StorageClass(copying: _storage)
    }
    return _storage
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Tfprof_OpLogEntry) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_storage, other_storage) in
        if _storage._name != other_storage._name {return false}
        if _storage._floatOps != other_storage._floatOps {return false}
        if _storage._types != other_storage._types {return false}
        if _storage._codeDef != other_storage._codeDef {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Tfprof_OpLog: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "log_entries"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Tfprof_OpLog) -> Bool {
    if self.logEntries != other.logEntries {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
