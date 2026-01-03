#[cfg(test)]
mod remote_workflow_integration_tests {
    use r#gen::{
        commands::remote::{RemoteCommand, handle_remote_command},
        operation_management::push,
        test_helpers::setup_gen,
    };
    use gen_models::operations::{Branch, Defaults, Remote};

    /// Test remote deletion with branch associations (should set to null)
    #[test]
    fn test_remote_deletion_with_branch_associations() {
        let context = setup_gen();
        let op_conn = context.operations().conn();

        // Add remotes
        Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
        Remote::create(op_conn, "upstream", "https://genhub.bio/upstream/repo.gen").unwrap();

        // Create branches and associate with remotes
        let main_branch = Branch::get_by_name(op_conn, "main").unwrap();
        let feature_branch = Branch::get_or_create(op_conn, "feature");
        let develop_branch = Branch::get_or_create(op_conn, "develop");

        Branch::set_remote(op_conn, main_branch.id, Some("origin")).unwrap();
        Branch::set_remote(op_conn, feature_branch.id, Some("origin")).unwrap();
        Branch::set_remote(op_conn, develop_branch.id, Some("upstream")).unwrap();

        // Verify initial associations
        assert_eq!(
            Branch::get_remote(op_conn, main_branch.id),
            Some("origin".to_string())
        );
        assert_eq!(
            Branch::get_remote(op_conn, feature_branch.id),
            Some("origin".to_string())
        );
        assert_eq!(
            Branch::get_remote(op_conn, develop_branch.id),
            Some("upstream".to_string())
        );

        // Delete origin remote
        let remove_cmd = RemoteCommand::Remove {
            name: "origin".to_string(),
        };
        assert!(handle_remote_command(op_conn, &remove_cmd).is_ok());

        // Verify remote was deleted
        assert!(Remote::get_by_name_optional(op_conn, "origin").is_none());
        assert!(Remote::get_by_name_optional(op_conn, "upstream").is_some());

        // Verify branch associations were set to null for deleted remote
        assert_eq!(Branch::get_remote(op_conn, main_branch.id), None);
        assert_eq!(Branch::get_remote(op_conn, feature_branch.id), None);
        // Upstream association should remain intact
        assert_eq!(
            Branch::get_remote(op_conn, develop_branch.id),
            Some("upstream".to_string())
        );

        // Verify branches still exist
        assert!(Branch::get_by_id(op_conn, main_branch.id).is_some());
        assert!(Branch::get_by_id(op_conn, feature_branch.id).is_some());
        assert!(Branch::get_by_id(op_conn, develop_branch.id).is_some());
    }

    /// Test default remote deletion (should clear default)
    #[test]
    fn test_default_remote_deletion_clears_default() {
        let context = setup_gen();
        let op_conn = context.operations().conn();

        // Add remotes
        Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
        Remote::create(op_conn, "backup", "https://genhub.bio/backup/repo.gen").unwrap();

        // Set origin as default
        Defaults::set_default_remote(op_conn, Some("origin")).unwrap();
        assert_eq!(
            Defaults::get_default_remote(op_conn),
            Some("origin".to_string())
        );

        // Delete the default remote
        let remove_cmd = RemoteCommand::Remove {
            name: "origin".to_string(),
        };
        assert!(handle_remote_command(op_conn, &remove_cmd).is_ok());

        // Verify default was cleared
        assert_eq!(Defaults::get_default_remote(op_conn), None);
        assert_eq!(Defaults::get_default_remote_url(op_conn), None);
    }

    /// Test error scenarios across the remote workflow
    #[test]
    fn test_remote_workflow_error_scenarios() {
        let context = setup_gen();
        let op_conn = context.operations().conn();

        // Test 1: Try to associate branch with non-existent remote
        let main_branch = Branch::get_by_name(op_conn, "main").unwrap();
        let result = Branch::set_remote(op_conn, main_branch.id, Some("nonexistent"));
        assert!(
            result.is_err(),
            "Should fail when setting non-existent remote"
        );

        // Test 2: Try to set non-existent remote as default
        let set_default_cmd = RemoteCommand::SetDefault {
            name: "nonexistent".to_string(),
        };
        let result = handle_remote_command(op_conn, &set_default_cmd);
        assert!(
            result.is_err(),
            "Should fail when setting non-existent remote as default"
        );

        // Test 3: Try to remove non-existent remote
        let remove_cmd = RemoteCommand::Remove {
            name: "nonexistent".to_string(),
        };
        let result = handle_remote_command(op_conn, &remove_cmd);
        assert!(
            result.is_err(),
            "Should fail when removing non-existent remote"
        );

        // Test 4: Try to add remote with duplicate name
        Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
        let add_duplicate_cmd = RemoteCommand::Add {
            name: "origin".to_string(),
            url: "https://genhub.bio/different/repo.gen".to_string(),
        };
        let result = handle_remote_command(op_conn, &add_duplicate_cmd);
        assert!(
            result.is_err(),
            "Should fail when adding remote with duplicate name"
        );

        // Test 5: Try to add remote with invalid name
        let add_invalid_name_cmd = RemoteCommand::Add {
            name: "invalid name with spaces".to_string(),
            url: "https://genhub.bio/user/repo.gen".to_string(),
        };
        let result = handle_remote_command(op_conn, &add_invalid_name_cmd);
        assert!(
            result.is_err(),
            "Should fail when adding remote with invalid name"
        );

        // Test 6: Try to add remote with invalid URL
        let add_invalid_url_cmd = RemoteCommand::Add {
            name: "test".to_string(),
            url: "not-a-valid-url".to_string(),
        };
        let result = handle_remote_command(op_conn, &add_invalid_url_cmd);
        assert!(
            result.is_err(),
            "Should fail when adding remote with invalid URL"
        );

        // Test 7: Try to push without default remote set
        let push_result = push(&context, None);
        assert!(
            push_result.is_err(),
            "Should fail when pushing without default remote"
        );

        // Test 8: Verify database consistency after errors
        let remotes = Remote::list_all(op_conn);
        assert_eq!(
            remotes.len(),
            1,
            "Should only have the one successfully added remote"
        );
        assert_eq!(remotes[0].name, "origin");

        assert_eq!(
            Defaults::get_default_remote(op_conn),
            None,
            "Default remote should not be set"
        );
        assert_eq!(
            Branch::get_remote(op_conn, main_branch.id),
            None,
            "Branch should not have remote set"
        );
    }
}
