// swift-tools-version:5.5
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TapMLSwift",
    products: [
        .library(
            name: "TapMLSwift",
            targets: ["LLMChatObjC", "TapMLSwift"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "LLMChatObjC",
            path: "Sources/ObjC",
            cxxSettings: [
                .headerSearchPath("../../tvm_home/include"),
                .headerSearchPath("../../tvm_home/3rdparty/dmlc-core/include"),
                .headerSearchPath("../../tvm_home/3rdparty/dlpack/include")
            ]
        ),
        .target(
            name: "TapMLSwift",
            dependencies: ["LLMChatObjC"],
            path: "Sources/Swift"
        )
    ],
    cxxLanguageStandard: .cxx17
)
