#include "../omni-bots/Container.hpp"
#include "../omni-bots/Connectors.hpp"

int main() {
    using namespace omnibot;


    // Create your connectors
    Connector* fileReader = fileConnector("reader", "read.txt", { FilePerms::Read });
    Connector* fileWriter = fileConnector("writer", "write.txt", { FilePerms::Write });

    // Create your interfaces
    Interface* read = interface("Read Interface", { fileReader });
    Interface* write = interface("Write Interface", { fileWriter });
    Interface* readWrite = interface("Read/Write Interface", {fileReader, fileWriter});

    // Create your container
    Container container1({bus("Read Bus", {read}), bus("Write Bus", {write})});


    // Alternate Configuration
    // Container container2({
    //     bus( "Read Bus", {
    //         interface( "Read Interface", {
    //             fileConnector(
    //                 "reader", 
    //                 "read.txt", 
    //                 {FilePerms::Read}
    //             ),

    //             /* ... */
    //         }),
    //     }),

    //     bus( "Write Bus", {
    //         interface( "Write Interface", {
    //             fileConnector(
    //                 "writer", 
    //                 "write.txt", 
    //                 {FilePerms::Write}
    //             ),

    //             /* ... */
    //         }),

    //         /* ... */
    //     }),

    //     /* ... */
    // });
};