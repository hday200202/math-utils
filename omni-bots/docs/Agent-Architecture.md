# OmniBots (Local Agents) Architecture Layout

### Connector and Interface Creation:
```cpp
/* ==================================================== //
    Interfaces receive connectors with permissions and
    use those permissions to build their valid command
    list. A permission that isn't granted produces no
    command, so denial is never a runtime condition the
    agent can observe or retry against.
// ==================================================== */

// ---- Read-only inbound mail -------------------------
// The mailbox is bound inside the connector. The agent
// never sees an address, only the command name.

Interface mail_in = Interface({
    Outlook_Connector("vendor-pricing@company.com", {
        Perm::ReadUnread,
        Perm::MarkRead
    })
});

// Builds: [ read_unread, mark_read ]


// ---- Outbound mail, recipient pinned ----------------
// Perm::SendTo takes its destination at config time.
// Grant it twice and you get two distinct commands.

Interface mail_out = Interface({
    Outlook_Connector("automation@company.com", {
        Perm::SendTo("finance@company.com"),
        Perm::SendTo("purchasing@company.com")
    })
});

// Builds: [ send_to_finance, send_to_purchasing ]


// ---- Multiple connectors in one interface -----------
// An interface is a capability grouping, not a 1:1 
// wrapper around a system. Group by what a job needs.

Interface pricing_records = Interface({
    SharePoint_Connector("/sites/Ops/Lists/PriceChanges", {
        Perm::ListRead,
        Perm::ItemCreate
    }),
    IFS_Connector("PURCHASE_PART_SUPPLIER", {
        Perm::Read
    })
});

// Builds: [ list_price_changes, create_price_change,
//           read_part_supplier ]


// ---- Same connector, different scopes ---------------
// Two interfaces over the same system with different
// permission sets. Whoever holds only the first cannot
// reach the write path at all.

Interface ifs_read = Interface({
    IFS_Connector("CUSTOMER_ORDER", { Perm::Read })
});

Interface ifs_write = Interface({
    IFS_Connector("CUSTOMER_ORDER", {
        Perm::Read,
        Perm::Update(Field::DeliveryDate)
    })
});
```

---

### Supply the Container With Your Interfaces:
```cpp
/* ==================================================== //
    A container is a permission scope. Interfaces are
    supplied at construction; every agent spawned into
    the container draws from that same set. Different
    permission scopes mean different containers.

    spawnAgent returns the container, not the agent.
    The agent object is never exposed to the caller.
// ==================================================== */

// ---- Single agent -----------------------------------

Container watcher = Container({ mail_in, pricing_records });

watcher.spawnAgent(
    "price-scraper",                    // name
    "Read unread vendor mail. For "     // prompt
    "each price change found, "         //
    "create a PriceChanges list "       //
    "item with part number, old "       //
    "price, new price, and "            //
    "effective date. Mark the mail "    //
    "read.",                            //
    onNewMail(mail_in)                  // trigger
);


// ---- Multiple agents, one scope ---------------------
// Agents sharing a container share its interfaces. 
// Names are how you address them afterward and how they 
// appear in logs.

Container ops = Container({ mail_in, pricing_records, mail_out });

ops.spawnAgent(
    "price-scraper",                    // name
    "Read unread vendor mail and "      // prompt
    "record any price changes.",        //
    onNewMail(mail_in)                  // trigger
)
.spawnAgent(
    "daily-digest",                     // name
    "Summarize price changes "          // prompt
    "recorded in the last 24 hours "    // 
    "and send the summary to "          //
    "purchasing.",                      //
    onSchedule("0 6 * * 1-5")           // trigger
)
.spawnAgent(
    "backlog-sweeper",                  // name
    "Find PriceChanges items older "    // prompt
    "than 30 days with no "             //
    "effective date and list them.",    //
    onDemand()                          // trigger
);


// ---- Separate scopes --------------------------------
// A container the scraper can't reach. Nothing in the
// scraper's container links to IFS writes, so no prompt
// it receives can produce one.

Container readers = Container({ mail_in, ifs_read });
Container writers = Container({ ifs_write });
```

---

### Interfaces and Connectors:
    ╭─Interface───────────╮           ╭─Connector───────────╮
    │ Connectors:   [...] │◄──────────│ Permissions:  [...] │
    │ Valid Cmds:   [...] │           │ Data Links:   [...] │
    ╰─────────────────────╯           ╰─────────────────────╯

- An interface will be a dynamically created, compiled, and linked library during agent configuration.
- Consists of a set of pre-defined connections, builds a list of valid commands based on a connection and it's permissions.

---

### Visualization of Container Program:
       ╭──┴───┴───┴──╮   ╭──┴───┴───┴──╮   ╭──┴───┴───┴──╮
    ╭──│ Interface A │───│ Interface B │───│ Interface C │──╮
    │  ╰─────┬─┬─────╯   ╰─────┬─┬─────╯   ╰─────┬─┬─────╯  │
    │        ↓ ↑               ↓ ↑               ↓ ↑        │
    │  ╭─────┴─┴───────────────┴─┴───────────────┴─┴─────╮  │
    │  │                   Command Bus                   │  │
    │  ╰───────────────────────┬─┬───────────────────────╯  │
    │                          ↓ ↑                          │
    │                       ╭──┴─┴──╮                       │
    │                       │ Agent │                       │
    │                       ╰───┬───╯                       │
    │                           ↑                           │
    │                    ╭──────┴──────╮                    │
    ╰────────────────────│   Trigger   │────────────────────╯
                         ╰─────────────╯

- The agent recieves all of it's available commands and a task as an input prompt.
- Optional trigger to inform the agent it should perform its task.
- Agent will output commands from its list of valid commands to perform its task.
- Commands will go into the Command Bus, a queue that routes to the command's corresponding interface.