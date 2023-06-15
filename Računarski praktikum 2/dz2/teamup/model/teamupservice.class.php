<?php
    
    class TeamUpService {
        function authentication($user, $pass) {
            try {
                $db = DB::getConnection();
                $st = $db->prepare('SELECT * 
                                    FROM dz2_users 
                                    WHERE username = :user 
                                    AND password_hash = :pass;');
                $st->execute(array('user' => $user,
                                   'pass' => password_hash($pass, PASSWORD_DEFAULT)));
            } catch(PDOException $e) { 
                exit('PDO error [authentication]: ' . $e->getMessage()); 
            }

            $row = $st->fetch();
            if($row === false)
                return null;
            else
                return new Users($row['id']
                               , $row['username']
                               , $row['password_hash']
                               , $row['email']
                               , $row['registration_sequence']
                               , $row['has_registered']);
        }

        function getAllProjects() {
            try {
                $db = DB::getConnection();
                $st = $db->prepare('SELECT dz2_projects.*, username
                                    FROM dz2_projects, dz2_users
                                    WHERE dz2_projects.id_user = dz2_users.id;');  
                $st->execute();  
            } catch(PDOException $e) { 
                exit('PDO error [getAllProjects]: ' . $e->getMessage()); 
            }

            $arr = array();
            while($row = $st->fetch()) {
                $arr[] = array('project' => new Projects($row['id']
                                                       , $row['id_user']
                                                       , $row['title']
                                                       , $row['abstract']
                                                       , $row['number_of_members']
                                                       , $row['status']),
                               'author' => $row['username']);
            }

            return $arr;
        }

        function getMyProjects($user_id) {
            try {
                $db = DB::getConnection();
                $st = $db->prepare('SELECT dz2_projects.*, username, member_type
                                    FROM dz2_projects, dz2_users, dz2_members
                                    WHERE dz2_projects.id = dz2_members.id_project
                                    AND dz2_users.id = dz2_members.id_user
                                    AND dz2_members.id_user = :id;');  
                $st->execute(array('id' => $user_id));  
            } catch(PDOException $e) { 
                exit('PDO error [getMyProjects]: ' . $e->getMessage()); 
            }

            $arr = array();
            while($row = $st->fetch()) {
                $arr[] = array('project' => new Projects($row['id']
                                                       , $row['id_user']
                                                       , $row['title']
                                                       , $row['abstract']
                                                       , $row['number_of_members']
                                                       , $row['status']),
                               'author' => $row['username'],
                               'type'   => $row['member_type']);
            }

            return $arr;
        }

        function getProjectByID($project_id) {
            try {
                $db = DB::getConnection();
                $st = $db->prepare("SELECT username
                                         , title
                                         , abstract
                                         , number_of_members
                                         , COUNT(dz2_members.id_user) AS current_number_of_members
                                    FROM   dz2_projects
                                         , dz2_users
                                         , dz2_members
                                    WHERE dz2_projects.id_user = dz2_users.id
                                      AND dz2_projects.id = dz2_members.id_project
                                      AND member_type = 'member'
                                      AND dz2_projects.id = :id
                                    GROUP BY dz2_members.id_project;");
                $st->execute(array('id' => $project_id));
            } catch(PDOException $e) { 
                exit('PDO error [getProjectByID]: ' . $e->getMessage()); 
            }

            $arr = array();
            $row = $st->fetch(); //očekujem točno jedan redak!
            
            $arr['author_username']             = $row['username'];
            $arr['project_title']               = $row['title'];
            $arr['abstract']                    = $row['abstract'];
            $arr['number_of_members']           = $row['number_of_members'];
            $arr['current_number_of_members']   = $row['current_number_of_members'];

            return $arr;
        }

        function applicationForProject($project_id, $user_id) {
            try {
                $db = DB::getConnection();
                $st = $db->prepare("INSERT INTO dz2_members(id_project, id_user, member_type)
                                    VALUES (  :project_id
                                            , :user_id
                                            , 'member');");  
                $st->execute(array('project_id' => $project_id, 'user_id' => $user_id));
            } catch(PDOException $e) { 
                exit('PDO error [applicationForProject]: ' . $e->getMessage()); 
            }
        }

        function closeProject($project_id) {
            try {
                $db = DB::getConnection();
                $st = $db->prepare("UPDATE dz2_projects
                                    SET status = 'closed'
                                    WHERE id = :project_id;");
                $st->execute(array('project_id' => $project_id));
            } catch(PDOException $e) { 
                exit('PDO error [closeProject]: ' . $e->getMessage()); 
            }
        }

        function enterNewProject($user_id, $title, $abstract, $number, $status) {
            try {
                $db = DB::getConnection();

                $s1 = $db->prepare('INSERT INTO dz2_projects(id_user
                                                           , title
                                                           , abstract
                                                           , number_of_members
                                                           , status)
                                    VALUES (:id, :t, :a, :num, :st);');
                $s1->execute(array('id' => $user_id
                                 , 't'  => $title
                                 , 'a'  => $abstract
                                 , 'num'=> $number
                                 , 'st' => $status));

                $s2 = $db->prepare("INSERT INTO dz2_members(id_project, id_user, member_type)
                                    VALUES ((SELECT id_project
                                             FROM dz2_projects
                                             WHERE title = :t)
                                            , :user
                                            , 'author');");
                $s2->execute(array('t' => $title, 'user' => $user_id));
            } catch(PDOException $e) { 
                exit('PDO error [enterNewProject]: ' . $e->getMessage()); 
            }
        }

        function memberOf($user_id, $project_id) {
            try {
                $db = DB::getConnection();
                $st = $db->prepare('SELECT *
                                    FROM dz2_members
                                    WHERE id_project = :p
                                    AND id_user = :u;');
                $st->execute(array('p' => $project_id, 'u' => $user_id));
            } catch(PDOException $e) { 
                exit('PDO error [memberOf]: ' . $e->getMessage()); 
            }

            $number = 0;
            while($row = $st->fetch()) 
                $number = $number + 1;
            if($number > 0) return true;

            return false;
        }
    }
?>