import {useRef, useState} from "react";

export default function fileDrop() {

    const uploadRef = useRef(null)

    return(

        <div
            style={{
                width:'100%',
                display:'flex',
                justifyContent:'center',
                alignItems:'center',
                overflow:'hidden',
                background:'purple',
                cursor:'pointer',
                minHeight:'50%',

            }}
        >
            <input ref={uploadRef} type="file" style={{display:'none'}}/>

    </div>
        )
}