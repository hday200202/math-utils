/*
    watch-agent: a container that keeps running, printed as it goes.

    notes-agent proves a single command makes the round trip. This one is
    for watching the traffic: it loops ticks forever, drops a batch of work
    into the inbox before each one, and prints the bus window afterwards so
    every command and every result shows up in the console.

    The batch is the point. Each inbox message asks for several items at
    once, so a reply that carries three appends together - run in order off
    one generation - is visibly different from three replies carrying one
    each. Read the agent trace for which happened: the queues exist to make
    the first possible, and this is where you find out whether the model
    takes them up on it.

        ./bin/release/watch-agent          run until Ctrl-C
        ./bin/release/watch-agent 3        run three ticks and stop

    Two connectors so the routing is visible rather than implied:

        inbox   read only     where the next request arrives
        list    read, append  the shopping list being maintained

    The agent has no write_ or insert_ command for either file, because
    those permissions were never granted. It cannot empty the list even if
    something in the inbox tells it to - that is the capability model doing
    its job, and the History block in the window is where you can watch it
    refuse to have the option.
*/

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "../Container.hpp"
#include "../Connectors.hpp"
#include "../utils/File.hpp"

using namespace omnibot;


static const std::string INBOX = "inbox.txt";
static const std::string LIST  = "shopping-list.txt";

/*
    The work arrives in batches, not one item at a time.

    One item per tick only ever needs one append, so a reply carrying a
    single command is indistinguishable from a model that can only manage
    one. Asking for three in a message is what makes the difference visible:
    a reply either queues three appends together, executed in order off the
    one generation, or it does them across three turns and pays for three.
*/
static const std::vector<std::vector<std::string>> REQUESTS = {
    { "bananas", "coffee", "olive oil" },
    { "rice", "paper towels", "soap" },
    { "flour", "sugar", "butter", "salt" }
};


static std::string join(const std::vector<std::string>& items) {
    std::string out;
    for (const std::string& i : items) out += i + ", ";
    if (!out.empty()) out.resize(out.size() - 2);
    return out;
}


static void put(const std::string& path, const std::string& text) {
    files::File f;
    f.open(path, files::OpenMode::Write);
    f.write(text);
    f.close();
}


static std::string get(const std::string& path) {
    files::File f;
    f.open(path, files::OpenMode::Read);
    return f.isOpen() ? f.text() : std::string();
}


static void rule(const std::string& label) {
    std::cout << "\n================ " << label << " ================\n";
}


int main(int argc, char** argv) {
    /* 0 means run until interrupted */
    const size_t cycles = (argc > 1) ? std::stoul(argv[1]) : 0;

    put(LIST, "- milk\n- eggs\n");
    put(INBOX, "");

    Connector* inbox = fileConnector("inbox", INBOX, { FilePerms::Read });
    Connector* list  = fileConnector("list",  LIST,  { FilePerms::Read,
                                                       FilePerms::Append });

    Interface* shopping = interface("Shopping Interface", { inbox, list });

    Container ops({ shopping });

    /* echo every command the moment it is routed, not once the tick ends */
    CommandBus::trace(true);
    Agent::trace(true);

    /*
        0.6b is enough once the task carries a worked example. It was not
        enough before that, and 8b was no better - the failure was never
        model size, it was a format described instead of shown.
    */
    ops.model("models", "qwen3-0.6b")
       .spawnAgent(
            "shopper",
            /*
                What a dev would actually write. Nothing here names a
                command or describes the json - the dev does not know the
                command structure and should not have to. The line format
                is theirs to state, because it is a fact about their file.
            */
            "Read the inbox and add the items it asks for to the shopping "
            "list, one item per line, each line written as \"- item\\n\".",
            Trigger{}
        );

    std::cout << "watching. Ctrl-C to stop." << std::endl;

    for (size_t tick = 0; cycles == 0 || tick < cycles; ++tick) {
        const std::vector<std::string>& want = REQUESTS[tick % REQUESTS.size()];
        put(INBOX, "Please add these to the shopping list: " + join(want) + "\n");

        rule("tick " + std::to_string(tick) + "  inbox: " + join(want)
             + "  (" + std::to_string(want.size()) + " items)");
        std::cout.flush();

        ops.tick();

        /*
            The queues are drained by the time tick() returns, so what is
            worth reading here is the History block - every command the bus
            routed this episode, paired with what came back.
        */
        std::cout << ops.window();

        std::cout << "--- " << LIST << " ---\n" << get(LIST);
        std::cout.flush();

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
