/*
    notes-agent: the smallest end-to-end omni-bots run.

    One file, one connector, one interface, one bus, one agent, one tick.
    Nothing here is clever - the point is to watch a command make the whole
    trip and come back:

        model text -> extractCommands -> CommandBus -> Interface
                   -> FileConnector::executeCmd -> FileCmd::dump -> back out

    The agent is granted Read and Append on notes.txt and nothing else. It
    has no write_ or insert_ command because those permissions were never
    granted, so no prompt it receives can produce one - that is the whole
    argument for capabilities over policy, and this is the smallest place to
    see it hold.
*/

#include <iostream>

#include "../Container.hpp"
#include "../Connectors.hpp"
#include "../utils/File.hpp"

using namespace omnibot;


static const std::string NOTES = "notes.txt";


/* Read mode fails on a file that is not there, so seed it first. */
static void seedNotes() {
    files::File f;
    f.open(NOTES, files::OpenMode::Write);
    f.write("- milk\n- eggs\n");
    f.close();
}


static void showNotes(const std::string& when) {
    files::File f;
    f.open(NOTES, files::OpenMode::Read);
    std::cout << "\n--- " << when << " ---\n" << f.text() << "-------------\n";
}


int main() {
    seedNotes();
    showNotes("notes.txt before");

    /*
        Read and Append only. Comment Append out and re-run: the agent is
        never told append_notes exists, so it cannot emit one.
    */
    Connector* notesFile = fileConnector("notes", NOTES, {
        FilePerms::Read,
        FilePerms::Append
    });

    Interface* notes = interface("Notes Interface", { notesFile });

    Container ops({ notes });

    /*
        qwen3-8b, not qwen3-0.6b. The 0.6b loads in a second and is tempting
        for a smoke test, but given run()'s opening "Begin." it replies
        {"cmd": "done"} on turn one and the run ends having done nothing -
        which looks exactly like a plumbing failure and is not one. Anything
        8b and up reads the list first.
    */

    ops.model("models", "qwen3-8b")
       .spawnAgent(
            "notes-bot",
            "You are editing a shopping list. Read the list, then append one "
            "line for an item that is missing. Then you are finished.",
            Trigger{}
        );

    std::cout << "\nrunning...\n";
    ops.tick();

    showNotes("notes.txt after");

    return 0;
}
