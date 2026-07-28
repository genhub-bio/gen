#[cfg(test)]
mod remote_workflow_integration_tests {
    use r#gen::{
        commands::remote::{RemoteCommand, handle_remote_command},
        test_helpers::setup_gen,
    };
    use gen_models::operations::{Defaults, Remote, RemoteBranch};

    /// Test remote deletion with branch associations (should set to null)
    #[test]
    fn test_remote_deletion_with_branch_associations() {
        let context = setup_gen();
        let config_conn = context.config().conn();

        // Add remotes
        Remote::create(config_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
        Remote::create(
            config_conn,
            "upstream",
            "https://genhub.bio/upstream/repo.gen",
        )
        .unwrap();

        // Create branches and associate with remotes
        RemoteBranch::set_remote_validated(config_conn, "main", Some("origin")).unwrap();
        RemoteBranch::set_remote_validated(config_conn, "feature", Some("origin")).unwrap();
        RemoteBranch::set_remote_validated(config_conn, "develop", Some("upstream")).unwrap();

        // Verify initial associations
        assert_eq!(
            RemoteBranch::get_remote(config_conn, "main"),
            Some("origin".to_string())
        );
        assert_eq!(
            RemoteBranch::get_remote(config_conn, "feature"),
            Some("origin".to_string())
        );
        assert_eq!(
            RemoteBranch::get_remote(config_conn, "develop"),
            Some("upstream".to_string())
        );

        // Delete origin remote
        let remove_cmd = RemoteCommand::Remove {
            name: "origin".to_string(),
        };
        assert!(handle_remote_command(config_conn, &remove_cmd).is_ok());

        // Verify remote was deleted
        assert!(Remote::get_by_name_optional(config_conn, "origin").is_none());
        assert!(Remote::get_by_name_optional(config_conn, "upstream").is_some());

        // Verify branch associations were set to null for deleted remote
        assert_eq!(RemoteBranch::get_remote(config_conn, "main"), None);
        assert_eq!(RemoteBranch::get_remote(config_conn, "feature"), None);
        // Upstream association should remain intact
        assert_eq!(
            RemoteBranch::get_remote(config_conn, "develop"),
            Some("upstream".to_string())
        );
    }

    /// Test default remote deletion (should clear default)
    #[test]
    fn test_default_remote_deletion_clears_default() {
        let context = setup_gen();
        let config_conn = context.config().conn();

        // Add remotes
        Remote::create(config_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
        Remote::create(config_conn, "backup", "https://genhub.bio/backup/repo.gen").unwrap();

        // Set origin as default
        Defaults::set_default_remote(config_conn, Some("origin")).unwrap();
        assert_eq!(
            Defaults::get_default_remote(config_conn),
            Some("origin".to_string())
        );

        // Delete the default remote
        let remove_cmd = RemoteCommand::Remove {
            name: "origin".to_string(),
        };
        assert!(handle_remote_command(config_conn, &remove_cmd).is_ok());

        // Verify default was cleared
        assert_eq!(Defaults::get_default_remote(config_conn), None);
        assert_eq!(Defaults::get_default_remote_url(config_conn), None);
    }

    /// Test error scenarios across the remote workflow
    #[test]
    fn test_remote_workflow_error_scenarios() {
        let context = setup_gen();
        let config_conn = context.config().conn();

        // Test 1: Try to associate branch with non-existent remote
        let result = RemoteBranch::set_remote_validated(config_conn, "main", Some("nonexistent"));
        assert!(
            result.is_err(),
            "Should fail when setting non-existent remote"
        );

        // Test 2: Try to set non-existent remote as default
        let set_default_cmd = RemoteCommand::SetDefault {
            name: "nonexistent".to_string(),
        };
        let result = handle_remote_command(config_conn, &set_default_cmd);
        assert!(
            result.is_err(),
            "Should fail when setting non-existent remote as default"
        );

        // Test 3: Try to remove non-existent remote
        let remove_cmd = RemoteCommand::Remove {
            name: "nonexistent".to_string(),
        };
        let result = handle_remote_command(config_conn, &remove_cmd);
        assert!(
            result.is_err(),
            "Should fail when removing non-existent remote"
        );

        // Test 4: Try to add remote with duplicate name
        Remote::create(config_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
        let add_duplicate_cmd = RemoteCommand::Add {
            name: "origin".to_string(),
            url: "https://genhub.bio/different/repo.gen".to_string(),
        };
        let result = handle_remote_command(config_conn, &add_duplicate_cmd);
        assert!(
            result.is_err(),
            "Should fail when adding remote with duplicate name"
        );

        // Test 5: Try to add remote with invalid name
        let add_invalid_name_cmd = RemoteCommand::Add {
            name: "invalid name with spaces".to_string(),
            url: "https://genhub.bio/user/repo.gen".to_string(),
        };
        let result = handle_remote_command(config_conn, &add_invalid_name_cmd);
        assert!(
            result.is_err(),
            "Should fail when adding remote with invalid name"
        );

        // Test 6: Try to add remote with invalid URL
        let add_invalid_url_cmd = RemoteCommand::Add {
            name: "test".to_string(),
            url: "not-a-valid-url".to_string(),
        };
        let result = handle_remote_command(config_conn, &add_invalid_url_cmd);
        assert!(
            result.is_err(),
            "Should fail when adding remote with invalid URL"
        );

        // Test 7: Verify database consistency after errors
        let remotes = Remote::list_all(config_conn);
        assert_eq!(
            remotes.len(),
            1,
            "Should only have the one successfully added remote"
        );
        assert_eq!(remotes[0].name, "origin");

        assert_eq!(
            Defaults::get_default_remote(config_conn),
            None,
            "Default remote should not be set"
        );
        assert_eq!(
            RemoteBranch::get_remote(config_conn, "main"),
            None,
            "Branch should not have remote set"
        );
    }
}
