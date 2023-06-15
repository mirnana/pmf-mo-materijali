<?php require_once __SITE_PATH . '/view/_header.php'; ?>

<form method="POST" action="<?php echo __SITE_URL . '/index.php?rt=projects/getProjectByID'?>">
    <table>
        <tr><th>Naslov</th><th>Autor</th><th>Status</th>
        <?php
            foreach($projectList as $p) {
                echo '<tr>' .
                     '<td>' .
                        '<button type="submit"' .
                                'name="project_id"' .
                                'value="' . 
                                $p['project']->id . '">' . 
                                $p['project']->title . 
                        '</button>
                      </td>' .
                     '<td>' . $p['author']           . '</td>' .
                     '<td>' . $p['project']->status  . '</td>' .
                     '</tr>';
            }
        ?>
    </table>
</form>

<form method="POST" action="<?php echo __SITE_URL; ?>/index.php?rt=login/logout">
    <button type="submit">Odjava</button>
</form>

<?php require_once __SITE_PATH . '/view/_footer.php'; ?>