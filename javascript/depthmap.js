function get_depthmap_tab_index(){
    const [,...args] = [...arguments]
    return [get_tab_index('mode_depthmap'), ...args]
}
