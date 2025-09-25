* Add created_on to serialization/deserialization/structs so remotes have same created_on dates as local
* Handle missing fields in capnp wrappers gracefully
* can likely relax returning requirement of sqlite now we use hashes
* Fix prune_graph, maybe a good way to do it now is track created_on via BGEs to actually find the newest edges on a CI.
* use integer for created_on